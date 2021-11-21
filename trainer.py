import torch
from torch.utils.data import DataLoader
from PIL import Image

from abelian_actions import *
from _config import (
    _LOCAL_DATASET_BASEPATH,
)  # This is Dataset path ending such as ../celeba/img_align_celeba/img_align_celeba
from dataset import PairImageDataset

from tqdm import tqdm

# TODO : Make better pipeline... please..


def train():
    # load data
    batch_size = 64
    epochs = 10
    device = "cuda:1"
    tqdm_log = False
    types = "unet"

    train_data = PairImageDataset(_LOCAL_DATASET_BASEPATH)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=20
    )

    # load model
    model = PairAction(ngf=256, nz=128, im_size=64, action_type=types)
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004, betas=(0.5, 0.999))

    # train
    for epoch in range(epochs):
        if tqdm_log:
            pbar = tqdm(train_loader)
        else:
            pbar = train_loader

        total_loss = 0

        for i, data in enumerate(pbar):
            img1, img2 = data
            img1, img2 = img1.to(device), img2.to(device)

            loss = model(img1, img2)

            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            # print loss
            if (i + 1) % 2000 == 0:
                print(
                    f"Epoch [{epoch + 1}], Step [{i}], Loss: {total_loss / (i + 1):.4f}"
                )
                # save model
                torch.save(
                    model.state_dict(), f"./ckpts/model{types}_{epoch}_{i + 1}.pth"
                )


if __name__ == "__main__":
    train()
