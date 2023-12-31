import os
import sys
import random
import cv2
import time
import numpy as np 
import math
import torch 
import torch.nn as nn 
import torch_optimizer as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F 
import torch.distributed as dist
import threading
import queue

# from torch.cuda.amp import autocast, GradScaler

from model.accnet_w import ACC_UNet_W
from model.accnet import ACC_UNet
from model.accnet_lite import ACC_UNet_Lite
from model.multiresnet import MultiResUnet
from model.threeplusnet import UNet_3Plus

from dataset import myDataset

device = torch.device('cuda:1')
read_image_queue = queue.Queue(maxsize=8)
save_img_queue = queue.Queue(maxsize=8)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

dataset = myDataset('test')

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def normalize(img) :
    def custom_bend(x) :
        linear_part = x
        exp_positive = torch.pow( x, 1 / 4 )
        exp_negative = - torch.pow( -x, 1 / 4 )
        return torch.where(x > 1, exp_positive, torch.where(x < -1, exp_negative, linear_part))
    
    img = (img * 2) - 1
    img = custom_bend(img)
    img = torch.tanh(img)
    img = (img + 1) / 2
    return img

def rgb_to_hsl(rgb):
    # Ensure RGB values are in [0, 1]
    rgb = rgb.clamp(0, 1)

    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

    max_rgb, _ = torch.max(rgb, dim=1)
    min_rgb, _ = torch.min(rgb, dim=1)
    delta = max_rgb - min_rgb

    # Lightness calculation
    l = (max_rgb + min_rgb) / 2

    # Avoid division by zero; set delta to a small value if it is zero
    delta = torch.where(delta == 0, torch.full_like(delta, 1e-6), delta)

    # Saturation calculation
    s = torch.where(l == 0, torch.zeros_like(l), delta / (1 - torch.abs(2 * l - 1)))

    # Hue calculation
    hue = torch.zeros_like(delta)
    hue = torch.where((max_rgb == r) & (delta != 0), 60 * (((g - b) / delta) % 6), hue)
    hue = torch.where((max_rgb == g) & (delta != 0), 60 * (((b - r) / delta) + 2), hue)
    hue = torch.where((max_rgb == b) & (delta != 0), 60 * (((r - g) / delta) + 4), hue)

    hsl = torch.stack((hue, s, l), dim=1)
    return hsl

def rgb_to_yuv(rgb):
    """
    Convert an RGB image to YUV.
    """
    # Transformation matrix
    transform_matrix = torch.tensor([
        [ 0.299,  0.587,  0.114],
        [-0.147, -0.289,  0.436],
        [ 0.615, -0.515, -0.100]
    ], dtype=rgb.dtype, device=rgb.device)

    # Reshape the transform matrix to be compatible with the batch dimension
    transform_matrix = transform_matrix.unsqueeze(0)

    # Perform the transformation
    yuv = torch.tensordot(rgb, transform_matrix, dims=([1], [1]))

    # Adjust the dimensions to maintain (N, C, H, W) format
    # yuv = yuv.permute(0, 3, 1, 2)

    return yuv

def save_images(save_img_queue):
    while True:
        try:
            imgs = save_img_queue.get_nowait()
        except queue.Empty:
            time.sleep(1e-4)
            continue

        sample_before = imgs[0].cpu().detach().numpy().transpose(1, 2, 0)
        cv2.imwrite('test/01_before.exr', sample_before[:, :, :3], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        sample_after = imgs[1].cpu().detach().numpy().transpose(1, 2, 0)
        cv2.imwrite('test/02_after.exr', sample_after[:, :, :3], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        sample_current = imgs[2].cpu().detach().numpy().transpose(1, 2, 0)
        cv2.imwrite('test/03_output.exr', sample_current[:, :, :3], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])

save_thread = threading.Thread(target=save_images, args=(save_img_queue, ))
save_thread.daemon = True
save_thread.start()

def read_images(read_image_queue, dataset):
    while True:
        for batch_idx in range(len(dataset)):
            before, after = dataset[batch_idx]
            read_image_queue.put([before, after])

read_thread = threading.Thread(target=read_images, args=(read_image_queue, dataset))
read_thread.daemon = True
read_thread.start()

log_path = 'train_log'
num_epochs = 4444
lr = 1e-3
batch_size = 1

data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=False)
steps_per_epoch = data_loader.__len__()
print (f'steps per epoch: {steps_per_epoch}')

def get_learning_rate(step):
    if step < steps_per_epoch:
        mul = step / steps_per_epoch
        return lr * mul
    else:
        mul = np.cos((step - 2000) / (num_epochs * steps_per_epoch - 2000. ) * math.pi) * 0.5 + 0.5
        return (lr - 4e-7) * mul + 4e-7
    
# model = ACC_UNet_W(3, 3).to(device)
# model = ACC_UNet(3, 3).to(device)
# model = ACC_UNet_Lite(3, 3).to(device)
# model = MultiResUnet(3, 3).to(device)
model = UNet_3Plus(3, 3, is_batchnorm=False).to(device)
criterion_mse = nn.MSELoss()
criterion_l1 = nn.L1Loss()
optimizer = optim.Yogi(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, 'min')

before = None
after = None
outputs = None
rgb_output_masked = None
rgb_after_masked = None

step = 0
current_epoch = 0
# saved_batch_idx = 0

steps_loss = []
epoch_loss = []

try:
    checkpoint = torch.load('train_log/model_training.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print('loaded previously saved model')
except Exception as e:
    print (f'unable to load saved model: {e}')

try:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print ('loaded optimizer state')
except Exception as e:
    print (f'unable to load optimizer state: {e}')

try:
    step = checkpoint['step']
    print (f'step: {step}')
    current_epoch = checkpoint['epoch']
    print (f'epoch: {current_epoch + 1}')
    # saved_batch_idx = checkpoint['batch_idx']
    # print (f'saved batch index: {saved_batch_idx}')
except Exception as e:
    print (f'unable to set step and epoch: {e}')

try:
    steps_loss = checkpoint['steps_loss']
    print (f'loaded loss statistics for step: {step}')
    epoch_loss = checkpoint['epoch_loss']
    print (f'loaded loss statistics for epoch: {current_epoch + 1}')
except Exception as e:
    print (f'unable to load step and epoch loss statistics: {e}')


time_stamp = time.time()

epoch = current_epoch
start_timestamp = time.time()

while epoch < num_epochs + 1:

    for batch_idx in range(len(dataset)):
        time_stamp = time.time()
        before, after = read_image_queue.get()

        # if batch_idx < saved_batch_idx:
        #     continue
        # saved_batch_idx = 0

        # before, after = dataset[batch_idx]

    # random.seed()
    # for batch_idx, (before, after) in enumerate(data_loader):

        before = before.to(device, non_blocking = True)
        after = after.to(device, non_blocking = True)
        before = normalize(before).unsqueeze(0)
        after = normalize(after).unsqueeze(0)
        # before = before.to(memory_format=torch.channels_last)
        # after = after.to(memory_format=torch.channels_last)
        data_time = time.time() - time_stamp
        time_stamp = time.time()

        current_lr = get_learning_rate(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        optimizer.zero_grad(set_to_none=True)
        rgb_output = (model((before*2 -1)) + 1) / 2
        # rgb_output = (model(before) + 1) / 2

        rgb_before = before[:, :3, :, :]
        rgb_after = after[:, :3, :, :]

        # rgb_output_blurred = F.interpolate(rgb_output, scale_factor = 1 / 64, mode='bilinear', align_corners=False)
        # rgb_output_blurred = F. interpolate(rgb_output_blurred, scale_factor = 64, mode='bilinear', align_corners=False)
        # rgb_output_highpass = (rgb_output - rgb_output_blurred) + 0.5

        # rgb_after_blurred  = F.interpolate(rgb_after, scale_factor = 1 / 64, mode='bilinear', align_corners=False)
        # rgb_after_blurred = F.interpolate(rgb_after_blurred, scale_factor = 64, mode= 'bilinear', align_corners=False)
        # rgb_after_highpass = (rgb_after - rgb_after_blurred) + 0.5

        # rgb_before_blurred = F.interpolate(rgb_before, scale_factor = 1 / 64, mode='bilinear', align_corners=False)
        # rgb_before_blurred = F.interpolate(rgb_before_blurred, scale_factor = 64, mode='bilinear', align_corners=False)

        # loss = (rgb_output - rgb_after).abs().mean()
        # hsl_loss = criterion_mse(rgb_to_hsl(rgb_output), rgb_to_hsl(rgb_after))
        # yuv_loss = criterion_mse(rgb_to_yuv(rgb_output), rgb_to_yuv(rgb_after))
        # rgb_loss = criterion_mse(rgb_output, rgb_after)
        loss = criterion_mse(rgb_output, rgb_after)
        loss_l1 = criterion_l1(rgb_output, rgb_after)
        loss_l1_str = str(f'{loss_l1.item():.4f}')

        epoch_loss.append(float(loss_l1))
        steps_loss.append(float(loss_l1))

        loss.backward()
        optimizer.step()

        train_time = time.time() - time_stamp
        time_stamp = time.time()

        '''
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        '''

        if step % 20 == 1:
            sample_before = rgb_before[0].clone().cpu().detach().numpy().transpose(1, 2, 0)
            cv2.imwrite('test/01_before.exr', sample_before[:, :, :3], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
            sample_after = rgb_after[0].clone().cpu().detach().numpy().transpose(1, 2, 0)
            cv2.imwrite('test/02_after.exr', sample_after[:, :, :3], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
            sample_current = rgb_output[0].clone().cpu().detach().numpy().transpose(1, 2, 0)
            cv2.imwrite('test/03_output.exr', sample_current[:, :, :3], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])

            '''
            before_clone = before[0].clone().to('cpu', non_blocking = True)
            after_clone = after[0].clone().to('cpu', non_blocking = True)
            rgb_output_clone = rgb_output[0].clone().to('cpu', non_blocking = True)
            try:
                save_img_queue.put([before_clone, after_clone, rgb_output_clone], block=False)
            except:
                pass
            '''

        data_time += time.time() - time_stamp
        data_time_str = str(f'{data_time:.2f}')
        train_time_str = str(f'{train_time:.2f}')

        epoch_time = time.time() - start_timestamp
        days = int(epoch_time // (24 * 3600))
        hours = int((epoch_time % (24 * 3600)) // 3600)
        minutes = int((epoch_time % 3600) // 60)

        print (f'\rEpoch [{epoch + 1} - {days:02}d {hours:02}:{minutes:02}], Time:{data_time_str} + {train_time_str}, Batch [{batch_idx + 1} / {len(dataset)}], Lr: {optimizer.param_groups[0]["lr"]:.4e}, Loss L1: {loss_l1_str}', end='')
        # print (f'\rEpoch [{epoch + 1} / {num_epochs}], Time:{data_time:.2f} + {train_time:.2f}, Batch [{batch_idx + 1} / {len(data_loader)}], Lr: {optimizer.param_groups[0]["lr"]:.4e}, Loss L1: {loss_l1.item():.8f}', end='')
        step = step + 1

    torch.save({
        'step': step,
        'steps_loss': steps_loss,
        'epoch': epoch,
        'epoch_loss': epoch_loss,
        # 'batch_idx': batch_idx,
        'lr': optimizer.param_groups[0]['lr'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'train_log/model_training.pth')

    smoothed_loss = np.mean(moving_average(epoch_loss, 9))
    print(f'\rEpoch [{epoch + 1} - {days:02}d {hours:02}:{minutes:02}], Minimum L1 loss: {min(epoch_loss):.4f} Avg L1 loss: {smoothed_loss:.4f}, Maximum L1 loss: {max(epoch_loss):.4f}')
    scheduler.step(smoothed_loss)
    steps_loss = []
    epoch_loss = []
    epoch = epoch + 1
