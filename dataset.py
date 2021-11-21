import random
from glob import glob

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class PairImageDataset(Dataset):
    def __init__(self, image_path):
        self.image_paths = glob(image_path + "/*.jpg")
        self.transform = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.CenterCrop(128),
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
        ran = random.randint(0, len(self.image_paths) - 1)
        return (self._getimg(idx), self._getimg(ran))

