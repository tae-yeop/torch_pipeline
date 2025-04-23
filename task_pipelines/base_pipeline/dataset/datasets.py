import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import glob
import re
import random
import numpy as np
from itertools import permutations, combinations

##########################################
############### Loader ###################
##########################################
from PIL import Image, ImageFile

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# PNG의 경우 더 빨리 로딩하게끔
try:
    import pyspng
except ImportError:
    pyspng = None


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        if not img.mode == 'RGB':
            img = img.convert("RGB")
        return img
    
def cv_loader(path):
    image = cv2.imread(path) # np.array [H, W, C]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def png_loader(file_path):
    with open(file_path, 'rb') as f:
        image = pyspng.load(f.read())
    return image

loaders = {'pil': pil_loader, 'cv' : cv_loader, 'png' : png_loader}


from torchvision.transforms import InterpolationMode
interpolation = {"nearest": InterpolationMode.NEAREST,
                "box": InterpolationMode.BOX,
                "bilinear": InterpolationMode.BILINEAR,
                "bicubic": InterpolationMode.BICUBIC,
                "lanczos": InterpolationMode.LANCZOS}



class ImageDataset(Dataset):
    """
    가장 단순한 형태, 루트안에 모든 파일이 한번에 다 있다고 가정
    """
    def __init__(self, root, transforms=None):
        super().__init__()
        self.root = root
        self.file_list = [os.path.join(self.root, file_path) for file_path in os.listdir(self.root)]
        self.transformsn = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        img = Image.open(file).convert('RGB')

        # 흑백을 3차원처럼 처리
        if img.size(0) == 1:
            img = torch.cat((img, img, img), dim=0)
        if self.transforms is not None:
            img = self.transforms(img)

        return img
    

class FolderClassDataset(Dataset):
    """
    폴더별로 클래스 이미지들이 나눠져 있을 때
    예: root/class1/xxx.png, root/class2/yyy.jpg, ...
    """
    def __init__(self, root):
        self.root = root
        self.transform = transform

        classes = sorted(
            entry.name for entry in os.scandir(root) if entry.is_dir()
        ) # scandir : root 바로 아래에 있는 폴더와 파일 리스트를 준다

        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {root}")

        # 클래스명 -> 인덱스 매핑
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        # 이미지 경로와 해당 클래스 인덱스를 담을 리스트
        samples = []

        # 클래스 디렉토리를 순회하며 이미지 경로를 수집
        for cls_name in classes:
            cls_dir = os.path.join(root, cls_name)
            cls_idx = class_to_idx[cls_name]
            for entry in os.scandir(cls_dir):
                if entry.is_file() and entry.name.endswith(('jpg', 'jpeg', 'png')):
                    file_path = os.path.join(cls_dir, entry.name)
                    samples.append((file_path, cls_idx))


        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = pil_loader(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class FileNameClassDataset(Dataset):
    """
    이미지 파일 이름에 클래스가 있을 경우: 예) cat_123.jpg, dog_987.jpg 등
    파일명에서 클래스 정보를 추출해서 라벨로 사용하는 Dataset 예시입니다.

    예시:
        - root
            |- cat_123.jpg
            |- cat_234.jpg
            |- dog_345.jpg
            ...

    
    label_to_id는 {'cat': 0, 'dog': 1} 처럼 클래스명을 숫자로 매핑한 dict를 넣어주면 됩니다.
    """
    def __init__(self, root, image_transform=None, label_to_id=None):
        self.root = root
        self.file_list = [os.path.join(root, file_path) for file_path in os.listdir(root)
                          if os.path.isfile(os.path.join(root, file_path))]
        self.image_transform = image_transform


        if label_to_id is not None:
            self.label_to_id = label_to_id
        else:
            self.label_to_id = {'cat' : 0, 'dog': 1}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        img = Image.open(file).convert('RGB')

        # 흑백을 3차원처럼 처리
        if img.size(0) == 1:
            img = torch.cat((img, img, img), dim=0)
        if self.image_transform is not None:
            img = self.image_transform(img)

        # 파일 이름에서 클래스명 추출
        label_str = self.extract_label(fname)
        # 문자열 라벨 -> 정수 라벨 매핑
        try:
            label = self.label_to_id[label_str]
        except KeyError:
            raise KeyError(f"label_to_id에 '{label_str}' 항목이 없습니다. "
                           f"label_to_id={self.label_to_id}")

        return img, label

    def extract_label(self, fname):
        """
        파일 이름에서 레이블을 추출하는 메소드

        예: /path/to/cat_123.jpg --> 'cat'
            /path/to/dog_999.jpg --> 'dog'

        """
        filename = os.path.basename(fname)  # 'cat_123.jpg'
        match = re.match(r"^(.*?)_\d+\.\w+$", filename)
        if match:
            return match.group(1) # 'cat' or 'dog'
        else:
            return -1 # 레이블을 찾지 못한 경우 -1 반환

class ImageMoireDataset(Dataset):
    def __init__(self, root, transformation=None):
        super().__init__()
        self.root = root
        self.file_list = [os.path.join(self.root, file_path) for file_path in os.listdir(self.root)]
        self.transformation = transformation
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 손상된 이미지 처리
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}

        # 클린 이미지
        path_tar = self.file_list[idx]
        number = os.path.split(path_tar)[-1][0:4]
        # 모이레 이미지 얻기
        path_src = os.path.split(path_tar)[0] + '/' + os.path.split(path_tar)[-1][0:4] + '_moire.jpg'

        clean_img = pil_rgb_convert(Image.open(path_tar))
        moire_img = pil_rgb_convert(Image.open(path_src))
        clean_img, moire_img = self.transformation((clean_img, moire_img))
        
        data['clean_img'] = clean_img
        data['moire_img'] = moire_img
        return data


from transfomations import CenterCropMargin, Resize, RandomHorizontalFlip, Normalize

class MultiImageDataset(Dataset):
    def __init__(
        self,
        root_dir,
        sub_dir_1,
        sub_dir_2,
        sub_dir_3,
        loader_type='cv',
        interpolation_mode='bicubic',
        crop=False,
        resize_size=None,
        normalize=True,
        random_flip=0.5,
    ):
        super().__init__()
        self.root = root
        self.sub_dir_1_path = os.path.join(root, sub_dir_1)
        self.sub_dir_2_path = os.path.join(root, sub_dir_2)
        self.sub_dir_3_path = os.path.join(root, sub_dir_3)

        Image.init()

        self.sub_dir_1_fnames = sorted([f for f in os.listdir(self.src_path) if self._file_ext(f) in Image.EXTENSION])
        self.sub_dir_2_fnames = sorted([f for f in os.listdir(self.tgt_path) if self._file_ext(f) in Image.EXTENSION])
        self.sub_dir_3_fnames = sorted([f for f in os.listdir(self.cond_path) if self._file_ext(f) in Image.EXTENSION])

        if len(self.sub_dir_1_fnames)*len(self.sub_dir_2_fnames)*len(self.sub_dir_3_fnames) == 0:
            raise IOError('No image files found in the specified path')

        assert len(self.sub_dir_1_fnames) == len(self.sub_dir_2_fnames) == len(self.sub_dir_3_fnames)

        self.loader = loaders[loader_type]
        self.interpolation = interpolation[interpolation_mode]

        self.trsf_list = []
        if crop:
            self.crop = CenterCropMargin()
            self.trsf_list.append(self.crop)

        self.trsf_list.append(ToTensor())

        if resize_size is not None and interpolation != 'wo_resize':
            self.resizer = Resize(resize_size, interpolation=self.interpolation)
            self.trsf_list.append(self.resizer)

        if random_flip > 0:
            self.flipper = RandomHorizontalFlip(random_flip)
            self.trsf_list.append(self.flipper)

        if normalize:
            self.normalizer = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            self.trsf_list.append(self.normalizer)

        self.trsf = transforms.Compose(self.trsf_list)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        return len(self.sub_dir_1_fnames)

    def _load_raw_image(self, raw_idx):
        sub_dir_1_file_name = self.sub_dir_1_fnames[raw_idx]
        sub_dir_1_img = self.loader(os.path.join(self.sub_dir_1_path, sub_dir_1_file_name))

        sub_dir_2_file_name = self.sub_dir_2_fnames[raw_idx]
        sub_dir_2_img = self.loader(os.path.join(self.sub_dir_2_path, sub_dir_2_file_name))

        sub_dir_3_file_name = self.sub_dir_3_fnames[raw_idx]
        sub_dir_3_img = self.loader(os.path.join(self.sub_dir_3_path, sub_dir_3_file_name))

        data = {'sub_dir_1': sub_dir_1_img, 'sub_dir_2':sub_dir_2_img, 'sub_dir_3': sub_dir_3_img}

        return data

    def __getitem__(self, idx):
        data = self._load_raw_image(idx)
        if self.trsf is not None:
            data = self.trsf(data)
        return data



class MultiRandomDataset(Dataset):
    """
    root 폴더 안에:
          3412323_0.jpg
          3412323_1.jpg
          987654_0.jpg
          987654_1.jpg
    
    1) 파일 이름에서 id와 variation 정보를 추출
    2) id별로 variation들을 모아서, (variation_a, variation_b) 쌍을 미리 만들어둠
    3) getitem에서는 미리 만든 pairs에서 해당 index의 샘플을 불러와서 전달

    ** 이 경우 getitem에서 랜덤하게 뽑지 말자 => 미리 뽑아야할 전체 조합을 가지고 있어야 한다
    def __getitem__(self, idx):
        id_idx = idx // self.num_variations
        candidate = id_idx*self.num_variations + torch.randperm(self.num_variations)
        candidate = candidate[candidate!=idx]
        data1_sample = self.data[idx]
        data2_sample = self.data2[candidate[0]]

        return data1_sample, data2_sample
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # 1) id별로 파일 경로 모으기
        #    data_by_id = { id_number : [파일경로1, 파일경로2, ...], ... }
        self.data_by_id = {}
        for fname in os.listdir(root):
            # jpg(또는 jpeg)만 필터링하고 싶다면 조건 추가 가능
            full_path = os.path.join(root, fname)
            if not os.path.isfile(full_path):
                continue
            
            # "3412323_0.jpg" 형태 -> 정규표현식으로 ID, variation 추출
            match = re.match(r'^(\d+)_(\d+)\.\w+$', fname)

            if not match:
                # 파일명이 조건에 맞지 않는 경우는 건너뜀
                continue
            
            id_str, var_str = match.groups()  # 예: ("3412323", "0")
            id_num = int(id_str)
            # dictionary에 id_num 키가 없으면 초기화
            if id_num not in self.data_by_id:
                self.data_by_id[id_num] = []

            # 실제 파일 경로(full_path)를 저장
            self.data_by_id[id_num].append(full_path)
        

        # (2) ID별 variation 파일들을 이용해 (p1, p2, same_id) 쌍을 만들기
        self.samples = []

        for id_num, path_list in self.data_by_id.items():
            # permutations(path_list, 2)는 같은 id 내에서 (p1, p2) 모든 순서쌍
            # 만약 순서가 중요치 않다면 combinations(path_list, 2) 사용
            for p1, p2 in permutations(path_list, 2):
                # (이미지경로1, 이미지경로2, 해당 id)
                self.samples.append((p1, p2, id_num))


        # 필요하다면 전체 쌍을 무작위로 섞기
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # 미리 만들어둔 samples에서 가져오기
        path1, path2, id_num = self.samples[idx]

        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, id_num
    

dataset_dict = {
    'image': ImageDataset,
    'multi_image': MultiImageDataset,
}
