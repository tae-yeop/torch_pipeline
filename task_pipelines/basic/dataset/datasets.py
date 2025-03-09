import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageFile
import os
import glob
import random


dataset_dict = {
    'image' : ImageDataset
}


class ImageDataset(Dataset):
    def __init__(self, gt_file_list, transformation=None):
        super().__init__()
        self.gt_file_list = gt_file_list
        self.transformation = transformation
        
    def __len__(self):
        return len(self.gt_file_list)

    def __getitem__(self, idx):
        # 손상된 이미지 처리
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}

        # 클린 이미지
        path_tar = self.gt_file_list[idx]
        number = os.path.split(path_tar)[-1][0:4]
        # 모이레 이미지 얻기
        path_src = os.path.split(path_tar)[0] + '/' + os.path.split(path_tar)[-1][0:4] + '_moire.jpg'

        clean_img = pil_rgb_convert(Image.open(path_tar))
        moire_img = pil_rgb_convert(Image.open(path_src))
        clean_img, moire_img = self.transformation((clean_img, moire_img))
        
        data['clean_img'] = clean_img
        data['moire_img'] = moire_img
        return data