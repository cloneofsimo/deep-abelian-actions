import random
from glob import glob

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class PairImageDataset(Dataset):
    def __init__(self, image_path: str, im_size: int, n_imgs: int = 2):
        self.image_paths = glob(image_path + "/*.jpg")
        self.transform = transforms.Compose(
            [
                transforms.Resize(im_size),
                transforms.CenterCrop(im_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.n_imgs = n_imgs

    def __len__(self):
        return len(self.image_paths)

    def _getimg(self, idx):
        # open image
        img = Image.open(self.image_paths[idx])
        # transform image
        img = self.transform(img)
        return img

    def __getitem__(self, idx):
        if self.n_imgs == 2:
            ran = random.randint(0, len(self.image_paths) - 1)
            return (self._getimg(idx), self._getimg(ran))
        elif self.n_imgs == 3:
            ran = random.randint(0, len(self.image_paths) - 1)
            ran2 = random.randint(0, len(self.image_paths) - 1)
            return (self._getimg(idx), self._getimg(ran), self._getimg(ran2))


class ImageDataset(Dataset):
    def __init__(self, image_path: str, im_size: int):
        self.image_paths = glob(image_path + "/*.jpg")
        self.transform = transforms.Compose(
            [
                transforms.Resize(im_size),
                transforms.CenterCrop(im_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def _getimg(self, idx):
        # open image
        img = Image.open(self.image_paths[idx])
        # transform image
        img = self.transform(img)
        return img

    def __getitem__(self, idx):
        return self._getimg(idx)
