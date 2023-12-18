import os
import sys
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm

from torch.nn import functional as F

from pprint import pprint

from model.multiresnet import MultiResUnet
from model.threeplusnet import UNet_3Plus

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
    parser = argparse. ArgumentParser (description='Interpolation for a sequence of exr images')
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

    first_image = cv2.imread(input_files[0], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH).copy()
    h, w, _ = first_image.shape
    ph = ((h - 1) // 64 + 1) * 64
    pw = ((w - 1) // 64 + 1) * 64
    padding = (0, pw - w, 0, ph - h)

    output_folder = os.path.abspath(args.output)
    checkpoint = torch.load('train_log2/model2.pth')
    # checkpoint = torch.load('train_log/model.pth')

    device = torch.device('cuda')
    model = MultiResUnet(3, 3).to(device)
    # model = UNet_3Plus(3, 3, is_batchnorm=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    for input_file_path in input_files:
        print(input_file_path + ' ->>>>')
        img0 = cv2.imread(input_file_path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH).copy()
        img0 = torch.from_numpy(img0.copy()).permute (2, 0, 1)
        img0 = img0.cuda()
        img0 = normalize(img0)

        img0 = F.pad(img0, padding)

        input_tensor = img0.unsqueeze(0)
        input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)
            rgb_output = output
        
        # res_img = denormalize(rgb_output[0])
        res_img = rgb_output[0]
        res_img = res_img.cpu().detach().numpy().transpose(1, 2, 0)

        output_file_name = os.path.basename(input_file_path)
        output_file_path = os.path.join(output_folder, output_file_name)
        cv2.imwrite(output_file_path, res_img, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        print(output_file_path)