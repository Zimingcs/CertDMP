import argparse
import math
import os
import random
import socket
from statistics import multimode

import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from tqdm import tqdm

import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration
from utils.setup import get_model
import torch._dynamo

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Restoring Patch-attacked Images with Denoising Diffusion Models')
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument('--resume', default='ckpts/Cifar10_ddpm.pth.tar', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--grid_r", type=int, default=16,
                        help="Grid cell width r that defines the overlap between patches")
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps")
    parser.add_argument("--image_folder", default='results/images/', type=str,
                        help="Location to save restored images")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    parser.add_argument('--classifier', default='vit_base', type=str,
                        help='Base classifier')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def find_non_majority(arr):
    
    rows, cols = arr.shape
    major = list(map(multimode, arr))
    
    non_majority_elements = []
    
    for row in range(rows):
        non_majority_elements_col = []
        for col in range(cols):
            if (arr[row][col] != major[row][0]):
                
                non_majority_elements_col.append((row, col, arr[row, col]))
        if len(non_majority_elements_col) != 0:
            non_majority_elements.append(non_majority_elements_col)
    return non_majority_elements



def main():
    args, config = parse_args_and_config()
    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # create model
    print("=> creating denoising-diffusion model with wrapper...")
    #diffusion = DenoisingDiffusion(args, config)
    #model_denoiser = DiffusiveRestoration(diffusion, args, config)
    model_classifier = get_model(args.classifier, config.data.dataset, 'classifiers')

    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = datasets.__dict__[config.data.dataset](config)
    val_loader = DATASET.get_loaders()
    model_classifier = model_classifier.to(device)


    prediction_modify_list = []
    clean_prediction_list = []
    label_list = []
    patched_prediction_list = []
    with (torch.no_grad()):
        model_classifier.eval()
        for data, label in tqdm(val_loader):
            data = data.to(device)
            if config.data.dataset == 'Cifar10' or 'Imagenet':
                label = label.numpy()
            else:
                label = np.array(label[0])

            data_pred = data
            patched_output = model_classifier(data_pred)
            patched_conf, patched_pred = patched_output.max(1)
            patched_pred = patched_pred.detach().cpu().numpy()
            patched_prediction_list.append(patched_pred)
            prediction_modify_list.append(patched_pred.copy())
            label_list.append(label)

            pos = 0
            mask_size = config.sampling.mask_size
            # slide>=patch_size-1
            slide = config.sampling.mask_size - config.sampling.patch_size_ex - 1
            mask_num = math.ceil((224 - mask_size) / slide) + 1
            prediction_map = np.zeros([data.shape[0], mask_num * mask_num], dtype=int)


            for row in range(mask_num):
                row_pix = row * slide
                for column in range(mask_num):
                    column_pix = column * slide
                    mask = torch.ones_like(data_pred)
                    mask[:, :, row_pix:row_pix + mask_size, column_pix:column_pix + mask_size] = 0
                    masked_input = mask * data_pred

                    
                    masked_output = model_classifier(masked_input)
                    masked_conf, masked_pred = masked_output.max(1)
                    masked_pred = masked_pred.detach().cpu().numpy()
                    prediction_map[:, pos] = masked_pred
                    pos += 1
            
            minority_list = find_non_majority(prediction_map)

            for sub_list in minority_list:
                flag = False
                recover_output_pred_list = []
                for (m_row, m_col, label_s) in sub_list:
                    recover_input = torch.Tensor()
                    recover_input = recover_input.to(device)
                    
                    res = divmod(m_col, mask_num)
                    row = int(res[0] * slide)
                    column = int(res[1] * slide)
                    recover_slice = data[m_row:m_row+1, :, :, :]
                    
                    mask = []
                    for k in range(4):
                        mask.append(torch.zeros_like(recover_slice))
                    mask[0][:, :, 0:row + mask_size, 0:column + mask_size] = 1
                    mask[1][:, :, 0:row + mask_size, column:224] = 1
                    mask[2][:, :, row:224, column:column + 224] = 1
                    mask[3][:, :, row:224, 0:column + mask_size] = 1

                    for l in mask:
                        recover_input = torch.cat((l * recover_slice, recover_input), dim=0)

                   

                    recover_output = model_classifier(recover_input)
                    recover_output_conf, recover_output_pred = recover_output.max(1)
                    recover_output_pred = recover_output_pred.detach().cpu().numpy()

                
                    c = 0
                    for p in range(4):
                        if recover_output_pred[p] != label_s:
                            c = c + 1
                    if c == 4:
                        recover_output_pred_list.append((m_row, m_col, label_s))
               
                if len(recover_output_pred_list) == 1:
                    prediction_modify_list[-1][m_row] = label_s

                elif len(recover_output_pred_list) > 1:
                    prediction_modify_list[-1][m_row] = -1

            clean_acc = 0
            patched_acc = 0
            robust_acc = 0
            abstain = 0           
            for k, m, n in zip(label,
                               patched_pred,
                               prediction_modify_list[-1]):
                patched_acc += m == k
                robust_acc += n == k
                abstain += n == -1
            print("For one batch:\r\n Clean accuracy：{} Patch accuracy：{} Certified accuracy：{} Abstain: {}".format(clean_acc/len(label), patched_acc/len(label), robust_acc/len(label), abstain/len(label)))
            

    prediction_modify_list = np.concatenate(prediction_modify_list)
    patched_prediction_list = np.concatenate(patched_prediction_list)
    label_list = np.concatenate(label_list)

    clean_corr = 0
    robust = 0
    orig_corr = 0
    abstain_corr = 0
    for i, (prediction_map, label, orig_pred) in enumerate(zip(prediction_modify_list, label_list, patched_prediction_list)):

        orig_corr += orig_pred == label
        robust += prediction_map == label

    print(orig_corr/10000)
    print(robust/10000)
if __name__ == '__main__':
    main()
