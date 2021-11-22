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
    epochs = 50
    device = "cuda:1"

    action_type = "amw"
    loss_type = "simple"
    im_size = 64

    tqdm_log = True
    ckpt_freq = 1500
    affine_latent = False
    ident_preserving = True

    train_data = PairImageDataset(_LOCAL_DATASET_BASEPATH, im_size)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=20
    )

    # load model
    model = PairAction(
        ngf=256,
        nz=512,
        im_size=im_size,
        action_type=action_type,
        loss_type=loss_type,
        affine_latent=affine_latent,
        ident_preserving=ident_preserving,
    )
    # model.load_state_dict(torch.load("./ckpts/modelunet_2_2000.pth"))
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.1, verbose=True
    )

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

            if tqdm_log:
                pbar.set_description(
                    f"Epoch {epoch} | Loss: {loss.item():.4f} | Total Loss: {total_loss / (i + 1):.4f}"
                )
            # print loss
            if (i + 1) % ckpt_freq == 0:
                print(f"Epoch [{epoch}], Step [{i}], Loss: {total_loss / (i + 1):.4f}")
                # save model
                torch.save(
                    model.state_dict(),
                    f"./ckpts/model{action_type}_{loss_type}_{ident_preserving}_{epoch}_{i + 1}.pth",
                )
                scheduler.step(total_loss / (i + 1))


if __name__ == "__main__":
    train()
