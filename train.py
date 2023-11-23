import os
import sys
import cv2
import time
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import torch.distributed as dist

from model.accnet_w import ACC_UNet_W

from dataset import myDataset

device = torch.device('cuda')

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

log_path = 'train_log'
num_epochs = 4444
lr = 9e-6
batch_size = 4

dataset = myDataset()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
steps_per_epoch = data_loader.__len__()
print (f'steps per epoch: {steps_per_epoch}')

def get_learning_rate(step):
    if step < 999:
        mul = step / 999
        return lr * mul
    else:
        mul = np.cos((step - 2000) / (num_epochs * steps_per_epoch - 2000. ) * np.math.pi) * 0.5 + 0.5
        return (lr - 4e-7) * mul + 4e-7
    
model = ACC_UNet_W(3, 3).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)

try:
    checkpoint = torch.load('train_log/model_training-pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print('loaded previously saved model')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
except:
    print ('unable to load saved model')

time_stamp = time.time()

before = None
after = None
outputs = None
rgb_output_masked = None
rgb_after_masked = None

step = 0

for epoch in range (num_epochs):
    epoch_loss = []
    random.seed()

    for batch_idx, (before, after) in enumerate(data_loader):
        before = before.to(device, non_blocking = True)
        after = after.to(device, non_blocking = True)
        before = normalize(before)
        after = normalize(after)
        data_time_int = time.time() - time_stamp
        time_stamp = time.time()

        current_lr = get_learning_rate(step)
        for param_group in optimizer.param_groups:
            param_group['Ir'] = current_lr

        rgb_output = model(before)

        rgb_before = before[:, :3, :, :]
        rgb_after = after[:, :3, :, :]

        rgb_output_blurred = F.interpolate(rgb_output, scale_factor = 1 / 64, mode='bilinear', align_corners=False)
        rgb_output_blurred = F. interpolate(rgb_output_blurred, scale_factor = 64, mode='bilinear', align_corners=False)
        rgb_output_highpass = (rgb_output - rgb_output_blurred) + 0.5

        rgb_after_blurred-F.interpolate(rgb_after, scale_factor = 1 / 64, mode='bilinear', align_corners=False)
        rgb_after_blurred=F.interpolate(rgb_after_blurred, scale_factor = 64, mode= 'bilinear', align_corners=False)
        rgb_after_highpass = (rgb_after - rgb_after_blurred) + 0.5

        rgb_before_blurred = F.interpolate(rgb_before, scale_factor = 1 / 64, mode='bilinear', align_corners=False)
        rgb_before_blurred = F.interpolate(rgb_before_blurred, scale_factor = 64, mode='bilinear', align_corners=False)

        loss = (rgb_output - rgb_after).abs().mean()
        epoch_loss.append(float(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_time_int = time.time() - time_stamp
        time_stamp = time.time()

        print (f'\rEpoch [{epoch + 1} / {num_epochs}], Time:{data_time_int:.2f} + {train_time_int:.2f}, Batch [{batch_idx + 1} / {len(data_loader)}], Lr: {optimizer.param_groups[0]["lr"]:.4e}, Loss: {loss.item():.4f}', end='')
        step = step + 1

        if step % 20 == 1:
            sample_before = ((before[0].cpu().detach().numpy().transpose(1,2,0)))
            cv2.imwrite('test/01_before.exr', sample_before[:,:,3], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
            sample_after = ((after[0].cpu().detach().numpy().transpose(1,2,0)))
            cv2.imwrite('test/02_after.exr', sample_after[:,:,3], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
            sample_current = ((rgb_output[0].cpu().detach().numpy().transpose(1,2,0)))
            cv2.imwrite('test/03_output.exr', sample_after[:,:,3], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])

    print(f'\r\nEpoch [{epoch + 1} / {num_epochs}], Minimum loss: {min(epoch_loss):.4f} Avg loss: {(sum(epoch_loss) / len(epoch_loss)):.4f}, Maximum loss: {max(epoch_loss):.4f}')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state dict': optimizer.state_dict(),
    }, f'train_log/model_trainig.pth')