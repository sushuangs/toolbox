import os
import glob
import random
import pickle

from data import data_utils

import numpy as np
import imageio
import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, config, name='', train=True, benchmark=False):
        self.config = config
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.scale = config.scale
        
        self.crop_size = 480
        self.stride = 240
        self.lr_crop_size = self.crop_size // self.scale
        self.lr_stride = self.stride // self.scale  # LR对应的步长
        self.thresh_size = 0
        
        self._set_filesystem(config.dir_data)
        self.bin_root = os.path.join(self.apath, 'bin')
        os.makedirs(self.bin_root, exist_ok=True)

        if config.ext.find('img') >= 0 or benchmark:
            self.images_hr, self.images_lr = self._scan()
        elif config.ext.find('sep') >= 0:
            self.images_hr, self.images_lr = self._load_or_generate_bin()

    def _scan(self):
        names_hr = sorted(glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0])))
        names_lr = []
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            names_lr.append(os.path.join(
                self.dir_lr, 'X{}/{}x{}{}'.format(
                    self.scale, filename, self.scale, self.ext[1]
                )
            ))
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('.png', '.png')
        
        self.bin_hr_dir = os.path.join(
            self.bin_root, 
            f'HR_crop{self.crop_size}_stride{self.stride}'
        )

        self.bin_lr_dir = os.path.join(
            self.bin_root, 
            f'LR_bicubic_X{self.scale}_crop{self.lr_crop_size}_stride{self.lr_stride}'
        )
    def _generate_bin_from_img(self, img_paths, save_dir, crop_size, stride):
        bin_paths = []
        for img_path in img_paths:
            # 仅当需要生成二进制时，才调用imageio.imread（首次/重置）
            need_crop = False
            # 先计算所有裁剪位置（提前判断是否有需要生成的文件）
            h_space, w_space = [], []
            # 临时读取一次原图，获取尺寸（仅首次/重置时执行）
            img = imageio.imread(img_path)
            h, w = img.shape[:2]
            
            if h >= crop_size:
                h_space = np.arange(0, h - crop_size + 1, stride)
                if h - (h_space[-1] + crop_size) > self.thresh_size:
                    h_space = np.append(h_space, h - crop_size)
            if w >= crop_size:
                w_space = np.arange(0, w - crop_size + 1, stride)
                if w - (w_space[-1] + crop_size) > self.thresh_size:
                    w_space = np.append(w_space, w - crop_size)
            
            # 遍历裁剪位置，生成二进制文件
            for y in h_space:
                for x in w_space:
                    bin_filename = f'{os.path.splitext(os.path.basename(img_path))[0]}_y{y}_x{x}.pt'
                    bin_path = os.path.join(save_dir, bin_filename)
                    # 文件不存在/重置时，裁剪并保存（仅此时调用imageio.imread后的img数据）
                    if not os.path.isfile(bin_path) or self.config.ext.find('reset') >= 0:
                        need_crop = True
                        sub_img = img[y:y+crop_size, x:x+crop_size, ...] if len(img.shape)==3 else img[y:y+crop_size, x:x+crop_size]
                        with open(bin_path, 'wb') as f:
                            pickle.dump(sub_img, f)
                        print(f'Generated bin file: {bin_path}')
                    bin_paths.append(bin_path)
            # 释放临时img内存
            del img
        return bin_paths

    def _load_or_generate_bin(self):
        os.makedirs(self.bin_hr_dir, exist_ok=True)
        os.makedirs(self.bin_lr_dir, exist_ok=True)
        
        ori_hr_paths, ori_lr_paths = self._scan()
        
        hr_bin_paths = []
        for ori_hr in ori_hr_paths:
            prefix = os.path.splitext(os.path.basename(ori_hr))[0]
            existing_bin = glob.glob(os.path.join(self.bin_hr_dir, f'{prefix}_y*_x*.pt'))
            if existing_bin and self.config.ext.find('reset') < 0:
                hr_bin_paths.extend(sorted(existing_bin))
            else:
                new_bin = self._generate_bin_from_img([ori_hr], self.bin_hr_dir, self.crop_size, self.stride)
                hr_bin_paths.extend(new_bin)
        
        lr_bin_paths = []
        for ori_lr in ori_lr_paths:
            prefix = os.path.splitext(os.path.basename(ori_lr))[0]
            existing_bin = glob.glob(os.path.join(self.bin_lr_dir, f'{prefix}_y*_x*.pt'))
            if existing_bin and self.config.ext.find('reset') < 0:
                lr_bin_paths.extend(sorted(existing_bin))
            else:
                new_bin = self._generate_bin_from_img([ori_lr], self.bin_lr_dir, self.lr_crop_size, self.lr_stride)
                lr_bin_paths.extend(new_bin)
        
        assert len(hr_bin_paths) == len(lr_bin_paths), f'HR bin({len(hr_bin_paths)}) != LR bin({len(lr_bin_paths)})'
        return hr_bin_paths, lr_bin_paths

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx]

        filename = os.path.splitext(os.path.basename(f_hr))[0]
        if self.config.ext.find('sep') >= 0:
            filename = filename.split('_y')[0]

        if self.config.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)
        elif self.config.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)

        return lr, hr, filename

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = data_utils.set_channel(*pair, n_channels=self.config.n_colors)
        pair_t = data_utils.np2Tensor(*pair, rgb_range=self.config.rgb_range)
        return pair_t[0], pair_t[1], filename

    def get_patch(self, lr, hr):
        scale = self.scale
        if self.train:
            lr, hr = data_utils.get_patch(
                lr, hr,
                patch_size=self.config.patch_size,
                scale=scale,
            )
            if not self.config.no_augment:
                lr, hr = data_utils.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]
        return lr, hr

    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx