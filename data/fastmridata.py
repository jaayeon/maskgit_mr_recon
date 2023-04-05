import os, glob
from turtle import forward
import torch
import random
import pickle
import numpy as np
from typing import Sequence

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from util.mri_tools import rfft2, rifft2

''' FastMRIDataset
input: kspace
output: subsampled image

load kspace
-> to tensor (complex stack in channel)
-> subsample kspace
-> ifft to image
-> absolute & Root-Sum-of-Square if multicoil data
-> normalize(mean=0, std=1) * clamp(-6,6)
'''

''' FastMRIDataset
input: kspace
output: subsampled kspace

load kspace
-> to tensor (complex stack in channel)
-> subsample kspace
(-> normalize) ???
'''


class FastMRIDataset(Dataset):
    def __init__(self, opt, mode='train') -> None:
        opt.input_size=320
        self.opt = opt
        self.mode = mode
        self.domain = opt.domain

        #downsample
        self.down = opt.down

        self.rng = np.random.RandomState(opt.seed)
        self.datalist = sorted(glob.glob(os.path.join(opt.data_path, opt.dataset, mode, '*.pkl')))
        if mode=='train':
            self.datalist.extend(glob.glob(os.path.join(opt.data_path, opt.dataset, 'valid', '*.pkl')))
        
        self.do_downsample = True if opt.downsample>1 else False
        num_low_freqs = int(opt.input_size/opt.downsample*opt.low_freq_ratio)
        num_high_freqs = int(opt.input_size/opt.downsample)-num_low_freqs

        # if self.down == 'random':
        #     self._init_random_pattern()


    def __getitem__(self, idx):
        with open(self.datalist[idx], 'rb') as f:
            kdata = pickle.load(f)
        kdata = np.array(kdata) #320, 320, 2
        # kdata = self.to_tensor(kdata)
        kdata = torch.tensor(kdata).permute(2,0,1)
        kdata = self.scale(kdata)

        if self.do_downsample:
            down_kdata, mask = self.downsample(kdata, idx) #mask shape: [1,h,1]
            ssl_mask_2d = self.mk_ssl_mask(mask=mask, shape=kdata.shape)
            if self.domain=='img':
                down_img = rifft2(down_kdata, permute=True)
                full_img = rifft2(kdata, permute=True)

                down_img, mean, std = self.normalize_instance(down_img)
                full_img = self.normalize(full_img, mean, std)

                return {'down': down_img, 'full': full_img, 'mask': ssl_mask_2d}
            elif self.domain=='kspace':
                return {'down': down_kdata, 'full': kdata, 'mask': ssl_mask_2d}
            else:
                raise NotImplementedError
        else:
            ssl_mask_2d = self.mk_ssl_mask(shape=kdata.shape)
            if self.domain=='img':
                full_img = rifft2(kdata, permute=True)
                full_img = self.normalize(full_img)
                return {'down': full_img, 'full': full_img, 'mask': ssl_mask_2d}
            elif self.domain=='kspace':
                return {'down': kdata, 'full': kdata, 'mask': ssl_mask_2d}
            else:
                raise NotImplementedError


    '''
    def normalize(self, arr, eps=1e-08): #[0,1] for spatial domain
        max = torch.max(arr)
        min = torch.min(arr)
        arr = (arr-min)/(max-min+eps)
        return arr
    '''

    def normalize(self, arr, mean, std, eps=1e-08):
        normalized = arr-mean)/(std+eps)
        return torch.clamp(normalized)

        
    def normalize_instance(self, arr: torch.Tensor, eps=1e-08):
        mean = arr.mean()
        std = arr.std()
        normalized = self.normalize(arr, mean, std, eps=eps)
        return normalized.clamp(-6,6), mean, std
    

    def scale(self, arr): #[-6~6] for kspace
        absmax = torch.max(torch.abs(arr))
        arr = arr/absmax*10
        return arr
    

    def to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        if np.iscomplexobj(arr):
            arr = np.stack((arr.real, arr.img))
        return torch.from_numpy(arr)
    

    def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
        return torch.view_as_complex(data).numpy()


    def mk_ssl_mask(self, mask=None, shape=(2,320,320)):
        c,h,w=shape
        #mask: 0 is remove, 1 is keep -> shuffle to 0 is keep, 1 is remove
        #mask_ = torch.zeros(mask.shape, device=mask.device)
        #mask_[mask==0]=1
        if mask is not None:
            mask_ = 1-mask
            ssl_mask_2d=torch.ones(c,h,w)*mask_
        else:
            ssl_mask_2d = torch.zeros(c,h,w)
        return ssl_mask_2d


    def downsample(self, arr, idx):
        c,h,w=arr.shape
        downsample = self.rng.uniform(size=1)[0]*self.opt.downsample
        num_low_freqs = int(self.opt.input_size/downsample*self.opt.low_freq_ratio)
        num_high_freqs = int(self.opt.input_size/downsample)-num_low_freqs

        #center_mask
        center_mask = np.zeros(h, dtype=np.float32)
        pad = (h - num_low_freqs + 1) // 2
        center_mask[pad : pad+num_low_freqs]=1
        # assert center_mask.sum() == num_low_freqs
        center_mask = self.reshape_mask(center_mask, arr.shape)

        #acceleration mask
        if self.down=='random': 
            prob = num_high_freqs/(h-num_low_freqs)
            accel_mask = self.rng.uniform(size=(h))<prob
            # accel_mask = self.accel_mask[idx,...]
            accel_mask = self.reshape_mask(accel_mask, arr.shape)
        elif self.down=='uniform':
            adjusted_accel = int((h-num_low_freqs)/(num_high_freqs))
            accel_mask = np.zeros(h, dtype=np.float32)
            accel_mask[0::adjusted_accel]=1
            accel_mask = self.reshape_mask(accel_mask, arr.shape)
            
        #apply mask
        mask = torch.max(center_mask, accel_mask)
        downsampled_data = arr*mask+0.0

        return downsampled_data, mask

    def _init_random_pattern(self):

        num_low_freqs = int(self.opt.input_size/self.opt.downsample*self.opt.low_freq_ratio)
        num_high_freqs = int(self.opt.input_size/self.opt.downsample)-num_low_freqs

        h = self.opt.input_size
        b = len(self.datalist)
        c = 2 if self.opt.domain=='kspace' else 1

        prob = num_high_freqs/(h-num_low_freqs)
        self.accel_mask = self.rng.uniform(size=(b,h))<prob


    def reshape_mask(self, mask:np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        """Reshape mask to desired output shape"""
        h = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = h #[1,h,1]
        
        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
    
    def __len__(self):
        return len(self.datalist)