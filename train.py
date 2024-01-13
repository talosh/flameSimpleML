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
        print ('dataset init')
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
        del reader
        self.h = 256
        self.w = 256
        self.frame_multiplier = (self.src_w // self.w) * (self.src_h // self.h) * 4

        self.frames_queue = queue.Queue(maxsize=12)
        self.frame_read_thread = threading.Thread(target=self.read_frames_thread)
        self.frame_read_thread.daemon = True
        self.frame_read_thread.start()

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
                    with open(target_file_path, 'rb') as tfp:
                        target_reader = MinExrReader(sfp)
                        target_image_data = target_reader.image.copy().astype(np.float32)
                except Exception as e:
                    print (e)

                if source_image_data is None or target_image_data is None:
                    time.sleep(timeout)
                    continue

                print (f'source shape {source_image_data.shape}')
                print (f'target shape {target_image_data.shape}')

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
        img0 = cv2.imread(os.path.join(self.clean_root, self.clean_files[shuffled_index]), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        img1 = cv2.imread(os.path.join(self.done_root, self.done_files[shuffled_index]), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        return img0, img1
    
    def __getitem__(self, index):
        img0, img1 = self.getimg(index)

        q = random.uniform(0, 1)
        if q < 0.5:
            img0 = cv2.resize(img0, (0,0), fx=0.5, fy=0.5)
            img1 = cv2.resize(img1, (0,0), fx=0.5, fy=0.5)
        # if q < 0.75:
        #    img0 = cv2.resize(img0, (0,0), fx=0.25, fy=0.25)
        #    img1 = cv2.resize(img1, (0,0), fx=0.25, fy=0.25)

        img0, img1 = self.crop(img0, img1, self.h, self.w)
        
        p = random.uniform(0, 1)
        if p < 0.25:
            img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
            img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
        elif p < 0.5:
            img0 = cv2.rotate(img0, cv2.ROTATE_180)
            img1 = cv2.rotate(img1, cv2.ROTATE_180)
        elif p < 0.75:
            img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)

        return img0, img1

def main():
    parser = argparse.ArgumentParser(description='Training script.')

    # Required argument
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')

    # Optional arguments
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--model_type', type=int, default=1, help='Model type (int): 1 - MultiresNet, 2 - Unet3++ (default: 1)')
    parser.add_argument('--warmup', type=float, default=1, help='Warmup epochs (float) (default: 1)')
    parser.add_argument('--pulse', type=float, default=9, help='Period in number of epochs to pulse learning rate (float) (default: 9)')
    parser.add_argument('--pulse_amplitude', type=float, default=10, help='Learning rate pulse amplitude (percentage) (default: 10)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the pre-trained model (optional)')
    parser.add_argument('--device', type=int, default=None, help='Graphics card index (default: 0)')

    args = parser.parse_args()

    dataset = myDataset(args.dataset_path)

    '''
    # Access arguments using args.learning_rate, args.model_type, etc.
    print(f"Dataset Path: {args.dataset_path}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Model Type: {args.model_type}")
    print(f"Warmup: {args.warmup}")
    print(f"Pulse: {args.pulse}")
    print(f"Pulse Amplitude: {args.pulse_amplitude}")
    '''

    # Rest of your training script...

    while True:
        time.sleep(1e-8)

if __name__ == "__main__":
    main()

