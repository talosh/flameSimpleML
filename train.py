import os
import sys
import random
import struct
import ctypes
import argparse
import importlib
import queue
import threading
import time

import flameSimpleML_framework
importlib.reload(flameSimpleML_framework)
from flameSimpleML_framework import flameAppFramework

from models.multires_v001 import Model as Model_01



fw = flameAppFramework()
try:
    import numpy as np
    import torch
except:
    if fw.site_packages_folder in sys.path:
        print ('unable to import numpy and pytorch')
        sys.exit()
    else:
        sys.path.append(fw.site_packages_folder)
        try:
            import numpy as np
            import torch
        except:
            print ('unable to import numpy and pytorch')
            sys.exit()
        if fw.site_packages_folder in sys.path:
            sys.path.remove(fw.site_packages_folder)

class BufferReader:
    '''A lightweight io.BytesIO object with convenience functions.
    
    Params
    ------
    data : bytes-like
        Bytes for which random access is required.
    
    '''

    def __init__(self, data):
        self.data = data
        self.len = len(data)
        self.off = 0

    def read(self, n):
        '''Read next `n` bytes.'''
        v = self.data[self.off:self.off+n]
        self.off += n
        return v

    def read_null_string(self):
        '''Read a null-terminated string.'''
        s = ctypes.create_string_buffer(self.data[self.off:]).value
        if s != None:
            s = s.decode('utf-8')
            self.off += len(s) + 1
        return s

    def peek(self):
        '''Peek next byte.'''
        return self.data[self.off]

    def advance(self, n):
        '''Advance offset by `n` bytes.'''
        self.off += n

    def nleft(self):
        '''Returns the number of bytes left to read.'''
        return self.len - self.off - 1
    
class MinExrReader:
    '''Minimal, standalone OpenEXR reader for single-part, uncompressed scan line files.

    This OpenEXR reader makes a couple of assumptions
     - single-part files with arbitrary number of channels,
     - no pixel data compression, and
     - equal channel types (HALF, FLOAT, UINT).

    These assumptions allow us to efficiently parse and read the `.exr` file. In particular
    we gain constant offsets between scan lines which allows us to read the entire image
    in (H,C,W) format without copying.

    Use `MinimalEXR.select` to select a subset of channels in the given order. `MinimalEXR.select`
    tries to be smart when copying is required and when views are ok.
    
    Based on the file format presented in
    https://www.openexr.com/documentation/openexrfilelayout.pdf

    Attributes
    ----------
    shape: tuple
        Shape of image in (H,C,W) order
    image: numpy.array
        Uncompressed image data.
    attrs: dict
        OpenEXR header attributes.
    '''


    def __init__(self, fp):
        self.fp = fp
        self.image = None
        self.shape = None

        self._read_header()
        self._read_image()

    def select(self, channels, channels_last=True):
        '''Returns an image composed only of the given channels.
        
        Attempts to be smart about memory views vs. memory copies.

        Params
        ------
        channels: list-like
            Names of channels to be extracted. Appearance in list
            also defines the order of the channels. 
        channels_last: bool, optional
            When true return image in (H,W,C) format.

        Returns
        -------
        image: HxWxC or HxCxW array
            Selected image data.
        '''
        H,C,W = self.shape
        ids = [self.channel_map[c] for c in channels]                
        if len(ids) == 0:
            img = np.empty((H,0,W), dtype=self.image.dtype)
        else:
            diff = np.diff(ids)
            sH = slice(0, H)
            sW = slice(0, W)
            if len(diff) == 0:
                # single channel select, return view
                sC = slice(ids[0],ids[0]+1)
                img = self.image[sH,sC,sW]
            elif len(set(diff)) == 1:
                # mutliple channels, constant offset between, return view
                # Careful here with negative steps, ie. diff[0] < 0:
                start = ids[0]
                step = diff[0]
                end = ids[-1]+diff[0]
                end = None if end < 0 else end                
                sC = slice(start,end,step)
                img = self.image[sH,sC,sW]
            else:
                # multiple channels not slicable -> copy mem
                chdata = [self.image[sH,i:i+1,sW] for i in ids]
                img = np.concatenate(chdata, 1)
        
        if channels_last:
            img = img.transpose(0,2,1)
        return img

    def _read_header(self):
        self.fp.seek(0)        
        buf = BufferReader(self.fp.read(10000))

        # Magic and version and info bits
        magic, version, b2, b3, b4 = struct.unpack('<iB3B', buf.read(8))
        assert magic == 20000630, 'Not an OpenEXR file.'
        assert b2 in (0, 4), 'Not a single-part scan line file.'
        assert b3 == b4 == 0, 'Unused flags in version field are not zero.'

        # Header attributes
        self.attrs = self._read_header_attrs(buf)

        # Parse channels and datawindow
        self.compr = self._parse_compression(self.attrs)        
        self.channel_names, self.channel_types = self._parse_channels(self.attrs)
        self.channel_map = {cn:i for i,cn in enumerate(self.channel_names)}
        H, W = self._parse_data_window(self.attrs)
        self.shape = (H,len(self.channel_names),W)
        self.first_offset = self._read_first_offset(buf)
        
        # Assert our assumptions
        assert self.compr == 0x00, 'Compression not supported.'
        assert len(set(self.channel_types)) <= 1, 'All channel types must be equal.'

    def _read_image(self):
        # Here is a shortcut: We assume all channels of the same type and thus constant offsets between
        # scanlines (SOFF). Note, each scanline has a header (y-coordinate (int4), data size DS (int4)) and data in scanlines
        # is stored consecutively for channels (in order of appearance in header). Thus we can interpret the content
        # as HxCxW image with strides: (SOFF,DS*W,DS)
        H,C,W = self.shape

        if np.prod(self.shape) == 0:
            return np.empty(self.shape, dtype=np.float32)

        dtype  = self.channel_types[0]
        DS = np.dtype(dtype).itemsize
        SOFF = 8+DS*W*C        
        strides = (SOFF, DS*W, DS)
        nbytes = SOFF*H

        self.fp.seek(self.first_offset, 0)
        image = np.frombuffer(self.fp.read(nbytes), dtype=dtype, count=-1, offset=8)
        self.image = np.lib.stride_tricks.as_strided(image, (H,C,W), strides)

    def _read_header_attrs(self, buf):
        attrs = {}
        while buf.nleft() > 0:
            attr = self._read_header_attr(buf)
            if attr is None:
                break
            attrs[attr[0]] = attr
        return attrs

    def _read_header_attr(self, buf):
        if buf.peek() == 0x00:
            buf.advance(1)
            return None
        aname = buf.read_null_string()
        atype = buf.read_null_string()
        asize = struct.unpack('<i', buf.read(4))[0]
        data = buf.read(asize)
        return (aname, atype, asize, data)

    def _parse_channels(self, attrs):
        attr = attrs['channels']
        assert attr[1] == 'chlist', 'Unexcepted type for channels attribute.'
        buf = BufferReader(attr[-1])
        channel_names, channel_types = [], []
        PT_LOOKUP = [np.uint32, np.float16, np.float32]
        while buf.nleft() > 0 and buf.peek() != 0x00:            
            channel_names.append(buf.read_null_string())
            pt = struct.unpack('<i', buf.read(4))[0]
            channel_types.append(PT_LOOKUP[pt])
            buf.advance(12) # skip remaining entries
        if buf.nleft() > 0:
            buf.advance(1) # account for zero byte
        return channel_names, channel_types

    def _parse_data_window(self, attrs):
        attr = attrs['dataWindow']
        assert attr[1] == 'box2i', 'Unexcepted type for dataWindow attribute.'
        xmin, ymin, xmax, ymax = struct.unpack('<iiii', attr[-1])
        return (ymax-ymin+1, xmax-xmin+1)

    def _parse_compression(self, attrs):
        return attrs['compression'][-1][0]

    def _read_offsets(self, buf):
        offsets = []
        while buf.nleft() > 0 and buf.peek() != 0x00:
            o = struct.unpack('<Q', buf.read(8))[0]
            offsets.append(o)
        if buf.nleft() > 0:
            buf.advance(1) # account for zero byte
        return offsets

    def _read_first_offset(self, buf):
        assert buf.nleft() > 0 and buf.peek() != 0x00, 'Failed to read offset.'
        return struct.unpack('<Q', buf.read(8))[0]

class myDataset(torch.utils.data.Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.source_root = os.path.join(self.data_root, 'source')
        self.target_root = os.path.join(self.data_root, 'target')
        self.source_files = [os.path.join(self.source_root, file) for file in sorted(os.listdir(self.source_root))]
        self.target_files = [os.path.join(self.target_root, file) for file in sorted(os.listdir(self.target_root))]
        self.indices = list(range(len(self.source_files)))

        try:
            with open(self.source_files[0], 'rb') as fp:
                reader = MinExrReader(fp)
        except Exception as e:
            print (f'Unable to read {self.source_files[0]}: {e}')
            sys.exit()

        self.src_h = reader.shape[0]
        self.src_w = reader.shape[2]
        self.in_channles = reader.shape[1]

        del reader

        try:
            with open(self.target_files[0], 'rb') as fp:
                reader = MinExrReader(fp)
        except Exception as e:
            print (f'Unable to read {self.source_files[0]}: {e}')
            sys.exit()

        self.src_h = reader.shape[0]
        self.src_w = reader.shape[2]
        self.out_channels = reader.shape[1]

        del reader
        self.h = 256
        self.w = 256
        self.frame_multiplier = (self.src_w // self.w) * (self.src_h // self.h) * 4

        self.frames_queue = queue.Queue(maxsize=4)
        self.frame_read_thread = threading.Thread(target=self.read_frames_thread)
        self.frame_read_thread.daemon = True
        self.frame_read_thread.start()

        self.last_shuffled_index = -1
        self.last_source_image_data = None
        self.last_target_image_data = None

    def read_frames_thread(self):
        timeout = 1e-8
        while True:
            for index in range(len(self.source_files)):
                source_file_path = self.source_files[index]
                target_file_path = self.target_files[index]
                source_image_data = None
                target_image_data = None
                try:
                    with open(source_file_path, 'rb') as sfp:
                        source_reader = MinExrReader(sfp)
                        source_image_data = source_reader.image.copy().astype(np.float32)
                        del source_reader
                    with open(target_file_path, 'rb') as tfp:
                        target_reader = MinExrReader(tfp)
                        target_image_data = target_reader.image.copy().astype(np.float32)
                        del target_reader
                except Exception as e:
                    print (e)

                if source_image_data is None or target_image_data is None:
                    time.sleep(timeout)
                    continue
                
                self.frames_queue.put([
                    np.transpose(source_image_data, (0, 2, 1)),
                    np.transpose(target_image_data, (0, 2, 1))
                ])

            time.sleep(timeout)

    def __len__(self):
        return len(self.source_files) * self.frame_multiplier
    
    def crop(self, img0, img1, h, w):
        np.random.seed(None)
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        return img0, img1

    def getimg(self, index):
        shuffled_index = self.indices[index // self.frame_multiplier]
        
        if shuffled_index != self.last_shuffled_index:
            self.last_source_image_data, self.last_target_image_data = self.frames_queue.get()
            self.last_shuffled_index = shuffled_index
        
        return self.last_source_image_data, self.last_target_image_data

    def __getitem__(self, index):
        img0, img1 = self.getimg(index)

        q = random.uniform(0, 1)
        if q < 0.5:
            img0, img1 = self.crop(img0, img1, self.h, self.w)
            img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
            img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        elif q < 0.75:
            img0, img1 = self.crop(img0, img1, self.h // 2, self.w // 2)
            img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
            img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
            img0 = torch.nn.functional.interpolate(img0.unsqueeze(0), scale_factor=2, mode='bilinear', align_corners=False)[0]
            img1 = torch.nn.functional.interpolate(img1.unsqueeze(0), scale_factor=2, mode='bilinear', align_corners=False)[0]
        else:
            img0, img1 = self.crop(img0, img1, int(self.h * 2), int(self.w * 2))
            img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
            img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
            img0 = torch.nn.functional.interpolate(img0.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False)[0]
            img1 = torch.nn.functional.interpolate(img1.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False)[0]
        
        p = random.uniform(0, 1)
        if p < 0.25:
            img0 = torch.flip(img0.transpose(1, 2), [2])
            img1 = torch.flip(img1.transpose(1, 2), [2])
        elif p < 0.5:
            img0 = torch.flip(img0, [1, 2])
            img1 = torch.flip(img1, [1, 2])
        elif p < 0.75:
            img0 = torch.flip(img0.transpose(1, 2), [1])
            img1 = torch.flip(img1.transpose(1, 2), [1])

        return img0, img1

def write_exr(image_data, filename, half_float = False, pixelAspectRatio = 1.0):
    height, width, depth = image_data.shape
    red = image_data[:, :, 0]
    green = image_data[:, :, 1]
    blue = image_data[:, :, 2]
    if depth > 3:
        alpha = image_data[:, :, 3]
    else:
        alpha = np.array([])

    channels_list = ['B', 'G', 'R'] if not alpha.size else ['A', 'B', 'G', 'R']

    MAGIC = 20000630
    VERSION = 2
    UINT = 0
    HALF = 1
    FLOAT = 2

    def write_attr(f, name, type, value):
        f.write(name.encode('utf-8') + b'\x00')
        f.write(type.encode('utf-8') + b'\x00')
        f.write(struct.pack('<I', len(value)))
        f.write(value)

    def get_channels_attr(channels_list):
        channel_list = b''
        for channel_name in channels_list:
            name_padded = channel_name[:254] + '\x00'
            bit_depth = 1 if half_float else 2
            pLinear = 0
            reserved = (0, 0, 0)  # replace with your values if needed
            xSampling = 1  # replace with your value
            ySampling = 1  # replace with your value
            channel_list += struct.pack(
                f"<{len(name_padded)}s i B 3B 2i",
                name_padded.encode(), 
                bit_depth, 
                pLinear, 
                *reserved, 
                xSampling, 
                ySampling
                )
        channel_list += struct.pack('c', b'\x00')

            # channel_list += (f'{i}\x00').encode('utf-8')
            # channel_list += struct.pack("<i4B", HALF, 1, 1, 0, 0)
        return channel_list
    
    def get_box2i_attr(x_min, y_min, x_max, y_max):
        return struct.pack('<iiii', x_min, y_min, x_max, y_max)

    with open(filename, 'wb') as f:
        # Magic number and version field
        f.write(struct.pack('I', 20000630))  # Magic number
        f.write(struct.pack('H', 2))  # Version field
        f.write(struct.pack('H', 0))  # Version field
        write_attr(f, 'channels', 'chlist', get_channels_attr(channels_list))
        write_attr(f, 'compression', 'compression', b'\x00')  # no compression
        write_attr(f, 'dataWindow', 'box2i', get_box2i_attr(0, 0, width - 1, height - 1))
        write_attr(f, 'displayWindow', 'box2i', get_box2i_attr(0, 0, width - 1, height - 1))
        write_attr(f, 'lineOrder', 'lineOrder', b'\x00')  # increasing Y
        write_attr(f, 'pixelAspectRatio', 'float', struct.pack('<f', pixelAspectRatio))
        write_attr(f, 'screenWindowCenter', 'v2f', struct.pack('<ff', 0.0, 0.0))
        write_attr(f, 'screenWindowWidth', 'float', struct.pack('<f', 1.0))
        f.write(b'\x00')  # end of header

        # Scan line offset table size and position
        line_offset_pos = f.tell()
        pixel_data_start = line_offset_pos + 8 * height
        bytes_per_channel = 2 if half_float else 4
        # each scan line starts with 4 bytes for y coord and 4 bytes for pixel data size
        bytes_per_scan_line = width * len(channels_list) * bytes_per_channel + 8 

        for y in range(height):
            f.write(struct.pack('<Q', pixel_data_start + y * bytes_per_scan_line))

        channel_data = {'R': red, 'G': green, 'B': blue, 'A': alpha}

        # Pixel data
        for y in range(height):
            f.write(struct.pack('I', y))  # Line number
            f.write(struct.pack('I', bytes_per_channel * len(channels_list) * width))  # Pixel data size
            for channel in sorted(channels_list):
                f.write(channel_data[channel][y].tobytes())
        f.close

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

def restore_normalized_values(image_array, torch = None):
    if torch is None:
        import torch

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

def main():
    parser = argparse.ArgumentParser(description='Training script.')

    # Required argument
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')

    # Optional arguments
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--type', type=int, default=1, help='Model type (int): 1 - MultiresNet, 2 - Unet3++ (default: 1)')
    parser.add_argument('--warmup', type=float, default=1, help='Warmup epochs (float) (default: 1)')
    parser.add_argument('--pulse', type=float, default=9, help='Period in number of epochs to pulse learning rate (float) (default: 9)')
    parser.add_argument('--pulse_amplitude', type=float, default=10, help='Learning rate pulse amplitude (percentage) (default: 10)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the pre-trained model (optional)')
    parser.add_argument('--device', type=int, default=0, help='Graphics card index (default: 0)')

    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.dataset_path, 'source')):
        print (f'dataset {args.dataset_path} must have "source" and "target" folders')
        sys.exit()
    if not os.path.isdir(os.path.join(args.dataset_path, 'target')):
        print (f'dataset {args.dataset_path} must have "source" and "target" folders')
        sys.exit()
    if not os.path.isdir(os.path.join(args.dataset_path, 'preview')):
        os.makedirs(os.path.join(args.dataset_path, 'preview'))

    read_image_queue = queue.Queue(maxsize=12)
    dataset = myDataset(args.dataset_path)

    def read_images(read_image_queue, dataset):
        while True:
            for batch_idx in range(len(dataset)):
                before, after = dataset[batch_idx]
                read_image_queue.put([before, after])

    read_thread = threading.Thread(target=read_images, args=(read_image_queue, dataset))
    read_thread.daemon = True
    read_thread.start()

    steps_per_epoch = dataset.__len__()

    device = torch.device(f'cuda:{args.device}')


    if args.type == 1:
        model = Model_01().get_training_model()(dataset.in_channles, dataset.out_channels).to(device)
    else:
        print (f'Model type {args.type} is not yet implemented')
        sys.exit()

    warmup_epochs = args.warmup
    pulse_dive = args.pulse_amplitude
    pulse_period = args.pulse
    lr = args.lr
    batch_size = 1

    criterion_mse = torch.nn.MSELoss()
    criterion_l1 = torch.nn.L1Loss()
    optimizer =torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    def warmup(current_step, lr = 4e-3, number_warmup_steps = 999):
        mul_lin = current_step / number_warmup_steps
        return lr * mul_lin if lr * mul_lin > 1e-111 else 1e-111

    train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch * pulse_period, eta_min = lr - (( lr / 100 ) * pulse_dive) )
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: warmup(step, lr=lr, number_warmup_steps=( steps_per_epoch * warmup_epochs ) / 10))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, train_scheduler], [steps_per_epoch * warmup_epochs])

    # Rest of your training script...

    step = 0
    current_epoch = 0
    preview_index = 0

    steps_loss = []
    epoch_loss = []

    if args.model_path:
        trained_model_path = args.model_path
        try:
            checkpoint = torch.load(trained_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print('loaded previously saved model')
        except Exception as e:
            print (f'unable to load saved model: {e}')
        try:
            start_timestamp = checkpoint.get('start_timestamp')
        except:
            start_timestamp = time.time()
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
    else:
        traned_model_name = 'flameSimpleML_model_' + fw.create_timestamp_uid()
        trained_model_dir = os.path.join(
            os.path.expanduser('~'),
            'flameSimpleML_models')
        if not os.path.isdir(trained_model_dir):
            os.makedirs(trained_model_dir)
        trained_model_path = os.path.join(trained_model_dir, traned_model_name)


    time_stamp = time.time()
    epoch = current_epoch

    while True:
        for batch_idx in range(len(dataset)):
            source, target = read_image_queue.get()
            source = source.to(device, non_blocking = True)
            target = target.to(device, non_blocking = True)

            source = normalize(source).unsqueeze(0) 
            target = normalize(target).unsqueeze(0)

            data_time = time.time() - time_stamp
            time_stamp = time.time()

            optimizer.zero_grad(set_to_none=True)
            output = model(source * 2 - 1)
            output = ( output + 1 ) / 2

            loss = criterion_mse(output, after)
            loss_l1 = criterion_l1(output, after)
            loss_l1_str = str(f'{loss_l1.item():.4f}')

            epoch_loss.append(float(loss_l1))
            steps_loss.append(float(loss_l1))

            loss.backward()

            current_lr = scheduler.get_last_lr()[0]
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.step()
            scheduler.step()

            train_time = time.time() - time_stamp
            time_stamp = time.time()

            if step % 40 == 1:
                preview_folder = os.path.join(args.dataset_path, 'preview')
                sample_source = source[0].clone().cpu().detach().numpy().transpose(1, 2, 0)
                sample_target = target[0].clone().cpu().detach().numpy().transpose(1, 2, 0)
                write_exr(sample_source, os.path.join(preview_folder, f'{preview_index:02}_source.exr'))
                write_exr(sample_target, os.path.join(preview_folder, f'{preview_index:02}_target.exr'))
                preview_index = preview_index + 1 if preview_index < 9 else 0

                # sample_current = rgb_output[0].clone().cpu().detach().numpy().transpose(1, 2, 0)

            data_time += time.time() - time_stamp
            data_time_str = str(f'{data_time:.2f}')
            train_time_str = str(f'{train_time:.2f}')
            current_lr_str = str(f'{scheduler.get_last_lr()[0]:.4e}')

            epoch_time = time.time() - start_timestamp
            days = int(epoch_time // (24 * 3600))
            hours = int((epoch_time % (24 * 3600)) // 3600)
            minutes = int((epoch_time % 3600) // 60)

            print (f'\033[K\rEpoch [{epoch + 1} - {days:02} d {hours:02}:{minutes:02}], Time:{data_time_str} + {train_time_str}, Batch [{batch_idx + 1} / {len(dataset)}], Lr: {current_lr_str}, Loss L1: {loss_l1_str}', end='')
            step = step + 1

if __name__ == "__main__":
    main()

