import os
import sys
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm

from print import print

from model.multiresnet import MultiResUnet

def normalize(img):
    def custom_bend(x):
        linear_part = x
        exp_positive = torch.pow( x, 1 / 4 )
        exp_negative = - torch.pow( -x, 1 / 4 )
        return torch.where(x > 1, exp_positive, torch.where(x < -1, exp_negative, linear_part))
    
    img = (img * 2) - 1
    img = custom_bend(img)
    img = torch.tanh(img)
    img = (img + 1) / 2
    return img

def denormalize(image_array):
    def custom_de_bend(x):
        linear_part = x
        inv_positive = torch.pow( x, 4 )
        inv_negative = -torch.pow( -x, 4 )
        return torch.where(x > 1, inv_positive, torch.where(x < -1, inv_negative, linear_part))

    epsilon = torch.tensor(4e-8, dtype=torch.float32).to(image_array.device)
    # clamp image befor arctanh
    image_array = torch.clamp((image_array * 2) - 1, -1.0 + epsilon, 1.0 - epsilon)
    # restore values from tanh  s-curve
    image_array = torch.arctanh(image_array)
    # restore custom bended values
    image_array = custom_de_bend(image_array)
    # move it to 0.0 - 1.0 range
    image_array = ( image_array + 1.0) / 2.0

    return image_array

if __name__ == '__main__':
    parser = argparse. Argumentarser (description='Interpolation for a sequence of exr images')
    parser.add_argument ('--input', dest='input', type=str, default=None)
    parser. add_argument('--output', dest='output', type=str, default=None)

    args = parser.parse_args()
    assert (not args.output is None or not args.input is None)

    img_formats = ['.exr',]
    files_list = []
    for f in os.listdir(args. input):
        name, ext = os.path.splitext(f)
        if ext in img_formats:
            files_list.append(f)

    input_files = []
    for file in sorted(files_list):
        input_file_path = os.path.join(args.input, file)
        if os.path.isfile(input_file_path):
            input_files.append(input_file_path)

    