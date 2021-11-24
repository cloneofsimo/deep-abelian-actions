import argparse
from hydra import compose, initialize

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from autoencoder import AutoEncoder
from dataset import ImageDataset

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
    nz = cfg.nz
    device = cfg.device
    epochs = cfg.epochs
    tqdm_log = cfg.tqdm_log
    ckpt_freq = cfg.ckpt_freq

    im_size = cfg.im_size
    action_type = cfg.action_type
    loss_type = cfg.loss_type
    affine_latent = cfg.affine_latent
    ident_preserving = cfg.ident_preserving
    debug = cfg.debug
    use_wandb = cfg.use_wandb
    ngf = cfg.ngf

    if use_wandb:
        import wandb
        attrs = cfg

        wandb_save_name = "-".join([f"{k}:{v}" for k, v in attrs.items()])

        if debug:
            wandb_save_name = "DEBUG" + wandb_save_name

        wandb.init(project = "DeepAbelian", entity = 'simoryu', name = wandb_save_name)

    train_data = ImageDataset(cfg.dataset_path, im_size)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    loss_func = nn.MSELoss()
    model = AutoEncoder(
        im_size=im_size,
        nz=nz,
        affine_latent=affine_latent,
        ngf=ngf,
        nc=3,
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
        pbar = tqdm(train_loader)
        total_loss = 0

        for i, img in enumerate(pbar):
            img = img.to(device)

            output = model(img)
            loss = loss_func(output, img)
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            pbar.set_description(
                f"Epoch {epoch} | Loss: {loss.item():.4f} | Total Loss: {total_loss / (i + 1):.4f}"
            )
            wandb.log({
                "loss": total_loss / (i+1)
            })
            # print loss
            if (i + 1) % ckpt_freq == 0 or i == 0:
                print(f"Epoch [{epoch}], Step [{i}], Loss: {total_loss / (i + 1):.4f}")
                # save model
                torch.save(
                    model.state_dict(),
                    f"./ckpts/{name}_model{action_type}_{loss_type}_{ident_preserving}_{epoch}_{i + 1}.pth",
                )
                scheduler.step(total_loss / (i + 1))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="euc_circles_ae", help="config yaml file", type=str
    )
    parser.add_argument("--debug", default="true")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)