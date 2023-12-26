import os
import sys
import random
import cv2
import time
import numpy as np 
import math
import torch 
import torch.nn as nn 
# import torch.optim as optim
import torch_optimizer as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F 
import torch.distributed as dist
import threading
import queue

from model.accnet_w import ACC_UNet_W
from model.accnet import ACC_UNet
from model.accnet_lite import ACC_UNet_Lite
from model.multiresnet import MultiResUnet
from model.threeplusnet import UNet_3Plus

from dataset import myDataset

import librosa
import soundfile as sf

# torch.cuda.set_device(1)
device = torch.device('cuda:0')
read_image_queue = queue.Queue(maxsize=8)
save_img_queue = queue.Queue(maxsize=8)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def fetch_segments(file_path, segment_duration, stride):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Convert durations to samples
    segment_length = int(segment_duration * sr)
    stride_length = int(stride * sr)

    # Initialize a list to store the segments
    segments = []

    # Loop through the file and extract segments
    for start in range(0, len(audio) - segment_length + 1, stride_length):
        end = start + segment_length
        segment = audio[start:end]
        segments.append(segment)

    return segments, sr

source_file_path = '/mnt/StorageMedia/dataset_audio/PaddySYNC.wav'
target_file_path = '/mnt/StorageMedia/dataset_audio/PaddySYNC_Cam.wav'

segment_duration = 1.0  # seconds
stride = 0.5  # seconds

source_audio_segments, sr = fetch_segments(source_file_path, segment_duration, stride)
target_audio_segments, sr = fetch_segments(target_file_path, segment_duration, stride)


log_path = 'train_log'
num_epochs = 4444
lr = 4e-3
batch_size = 1

steps_per_epoch = len(source_audio_segments)
print (f'steps per epoch: {steps_per_epoch}')

def get_learning_rate(step):
    if step < steps_per_epoch:
        mul = step / steps_per_epoch
        return lr * mul
    else:
        return lr
        # mul = np.cos((step - 2000) / (num_epochs * steps_per_epoch - 2000. ) * math.pi) * 0.5 + 0.5
        # return (lr - 4e-7) * mul + 4e-7

    
# model = ACC_UNet_W(3, 3).to(device)
# model = ACC_UNet(3, 3).to(device)
# model = ACC_UNet_Lite(3, 3).to(device)
model = MultiResUnet(3, 3).to(device)
# model = UNet_3Plus(3, 3, is_batchnorm=False).to(device)

criterion_mse = nn.MSELoss()
criterion_l1 = nn.L1Loss()
# optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
# optimizer = optim.SGD(model.parameters(), lr=lr)
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
    checkpoint = torch.load('train_log_audio/model_audio_training.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print('loaded previously saved model')
except Exception as e:
    print (f'unable to load saved model: {e}')

try:
    start_timestamp = checkpoint.get('start_timestamp')
except:
    start_timestamp = time.time()
'''
try:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print ('loaded optimizer state')
except Exception as e:
    print (f'unable to load optimizer state: {e}')
'''

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
while epoch < num_epochs + 1:
    random.seed()

    for batch_idx in range(len(source_audio_segments)):
        time_stamp = time.time()
        before = librosa.stft(source_audio_segments[batch_idx], center=False)
        after = librosa.stft(target_audio_segments[batch_idx], center=False)
        before = torch.from_numpy(before).float() / 255.
        after = torch.from_numpy(after).float() / 255.
        

        # if batch_idx < saved_batch_idx:
        #    continue
        # saved_batch_idx = 0

        # before, after = dataset[batch_idx]

        before = before.to(device, non_blocking = True)
        after = after.to(device, non_blocking = True)
        before = before.unsqueeze(0)
        after = after.unsqueeze(0)

        data_time = time.time() - time_stamp
        time_stamp = time.time()

        current_lr = get_learning_rate(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        optimizer.zero_grad(set_to_none=True)
        output = (model((before*2 -1)) + 1) / 2
        # rgb_output = (model(before) + 1) / 2

        # rgb_before = before[:, :3, :, :]
        # rgb_after = after[:, :3, :, :]

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
        loss = criterion_mse(output, after)
        loss_l1 = criterion_l1(output, after)
        loss_l1_str = str(f'{loss_l1.item():.4f}')

        epoch_loss.append(float(loss_l1))
        steps_loss.append(float(loss_l1))

        loss.backward()
        optimizer.step()

        train_time = time.time() - time_stamp
        time_stamp = time.time()
        
        if step % 40 == 1:
            sample_before = before[0].clone().cpu().detach().numpy()
            sample_after = after[0].clone().cpu().detach().numpy()
            sample_current = output[0].clone().cpu().detach().numpy()
            output_audio_before = librosa.istft(sample_before, center=False)
            output_audio_after = librosa.istft(sample_before, center=False)
            output_audio_current = librosa.istft(sample_before, center=False)
            sf.write('test_audio/01_before.wav', output_audio_before, sr)
            sf.write('test_audio/02_after.wav', output_audio_after, sr)
            sf.write('test_audio/03_output.wav', output_audio_current, sr)
            
        '''
        sample_before = before[0].to('cpu', non_blocking = True).detach().numpy().transpose(1, 2, 0)
        cv2.imwrite('test2/01_before.exr', sample_before[:, :, :3], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        sample_after = after[0].to('cpu', non_blocking = True).detach().numpy().transpose(1, 2, 0)
        cv2.imwrite('test2/02_after.exr', sample_after[:, :, :3], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        sample_current = rgb_output[0].to('cpu', non_blocking = True).detach().numpy().transpose(1, 2, 0)
        cv2.imwrite('test2/03_output.exr', sample_current[:, :, :3], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        '''

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
        step = step + 1

    torch.save({
        'step': step,
        'steps_loss': steps_loss,
        'epoch': epoch,
        'epoch_loss': epoch_loss,
        'start_timestamp': start_timestamp,
        # 'batch_idx': batch_idx,
        'lr': optimizer.param_groups[0]['lr'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'train_log_audio/model_audio_training.pth')
    
    smoothed_loss = np.mean(moving_average(epoch_loss, 9))
    epoch_time = time.time() - start_timestamp
    days = int(epoch_time // (24 * 3600))
    hours = int((epoch_time % (24 * 3600)) // 3600)
    minutes = int((epoch_time % 3600) // 60)
    print(f'\rEpoch [{epoch + 1} - {days:02}d {hours:02}:{minutes:02}], Minimum L1 loss: {min(epoch_loss):.4f} Avg L1 loss: {smoothed_loss:.4f}, Maximum L1 loss: {max(epoch_loss):.4f}')
    scheduler.step(smoothed_loss)
    steps_loss = []
    epoch_loss = []
    epoch = epoch + 1
