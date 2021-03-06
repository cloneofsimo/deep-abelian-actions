import argparse
from typing import Dict


from hydra import compose, initialize
from omegaconf import DictConfig

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


def train(args):
    initialize("config")
    print(args.config)
    print(args)
    cfg = compose(args.config)

    # params
    name = args.config
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    lr = cfg.lr
    device = cfg.device
    epochs = cfg.epochs
    tqdm_log = cfg.tqdm_log
    ckpt_freq = cfg.ckpt_freq

    nz = cfg.nz
    ngf = cfg.ngf

    im_size = cfg.im_size
    action_type = cfg.action_type
    loss_type = cfg.loss_type
    affine_latent = cfg.affine_latent
    ident_preserving = cfg.ident_preserving

    use_wandb = cfg.use_wandb

    dataset_path = cfg.dataset_path

    debug = cfg.debug
    if use_wandb:
        import wandb

        attrs = cfg

        # attrs["type"] = "NaiveFM"
        wandb_save_name = "-".join([f"{k}:{v}" for k, v in attrs.items()])

        if debug:
            wandb_save_name = "DEBUG" + wandb_save_name

        wandb.init(project="DeepAbelian", entity="simoryu", name=wandb_save_name)

    if loss_type in ["compat"]:
        n_imgs = 3
    else:
        n_imgs = 2

    train_data = PairImageDataset(dataset_path, im_size, n_imgs)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # load model
    model = PairAction(
        ngf=ngf,
        nz=nz,
        im_size=im_size,
        action_type=action_type,
        loss_type=loss_type,
        affine_latent=affine_latent,
        ident_preserving=ident_preserving,
    )

    # model.load_state_dict(torch.load("./ckpts/modelunet_2_2000.pth"))
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
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

            if n_imgs == 2:
                img1, img2 = data
                img1, img2 = img1.to(device), img2.to(device)

                loss = model(img1, img2)

            elif n_imgs == 3:
                img1, img2, img3 = data
                img1, img2, img3 = img1.to(device), img2.to(device), img3.to(device)

                loss = model(img1, img2, img3)

            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

            if tqdm_log:
                pbar.set_description(
                    f"Epoch {epoch} | Loss: {loss.item():.4f} | Total Loss: {total_loss / (i + 1):.4f}"
                )
            # print loss
            if (i + 1) % ckpt_freq == 0 or i == 0:
                print(f"Epoch [{epoch}], Step [{i}], Loss: {total_loss / (i + 1):.4f}")
                # save model
                torch.save(
                    model.state_dict(),
                    f"./ckpts/{name}_model{action_type}_{loss_type}_{ident_preserving}_{epoch}_{i + 1}.pth",
                )
                scheduler.step(total_loss / (i + 1))

            wandb.log({"loss": total_loss / (i + 1)})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="celeba", help="config yaml file", type=str)
    parser.add_argument("--debug", default="true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
