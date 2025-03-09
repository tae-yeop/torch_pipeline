import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import numbers

class RandomHorizontalFlipPair(transforms.RandomHorizontalFlip):
    def __call__(self, img_pair):
        if torch.randn(1) < self.p:
            return (TF.hflip(img_pair[0]), TF.hflip(img_pair[1]))
        return img_pair

class Resize(torch.nn.Module):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        self.size = size
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img_pair):
        return tuple(TF.resize(img, self.size, self.interpolation, self.max_size, self.antialias) for img in img_pair)

    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__ + '(size={0}, interpolation={1}, max_size={2}, antialias={3})'.format(
            self.size, interpolate_str, self.max_size, self.antialias)

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size

class RandomCrop(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = tuple(_setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))
    def forward(self, img_pair):
        # 이미지 크기와 타겟 크기를 기반으로 랜덤 위치 결정
        i, j, h, w = self.get_params(img_pair[0], self.size)
        # 결정된 위치로 두 이미지 모두 크롭
        return tuple(TF.crop(img, i, j, h, w) for img in img_pair)


    @staticmethod
    def get_params(img, output_size):
        _, h, w = TF.get_dimensions(img)
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
    

class Normalize(torch.nn.Module):
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, img_pair):
        img_pair = tuple(TF.normalize(img, self.mean, self.std, self.inplace) for img in img_pair)
        return img_pair
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ToTensor:
    def __call__(self, img_pair):
        return tuple(TF.to_tensor(img) for img in img_pair)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_transform(trans_type, train_img_size=512):
    if trans_type == 'UHDM_train':
        transform = transforms.Compose([
            Resize(train_img_size),
            RandomCrop(train_img_size),
            RandomHorizontalFlipPair(0.5),
            ToTensor(),
            # Normalize(0.5, 0.5)
        ])
    else:
        ...
    return transform