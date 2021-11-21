import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import ConvTranspose1d
import torchvision
import segmentation_models_pytorch as smp

class Upconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Upconv, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        

    def forward(self, x):
        x = self.main(x)
        return x

class SubDecoder(nn.Module):
    def __init__(self, nz, ngf, nc, im_size):
        super(SubDecoder, self).__init__()

        self.im_size = im_size
        self.upfactor = round(math.log2(im_size)) - 3
        assert self.upfactor > 0, "Image size must be greator than 8"

        

        self.up_size = round(2**(self.upfactor + 3))
        print("Decoder upsizing factor:", self.upfactor)
        print(f"img size {self.im_size}, up_size {self.up_size}")

        self.main = nn.Sequential(
            Upconv(nz, ngf * (2 ** self.upfactor), 4, 1, 0),
            *[
                Upconv(ngf * (2 ** mult), ngf * (2 ** (mult - 1)), 4, 2, 1) for mult in range(self.upfactor, 0, -1)
            ],
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
        )

    def forward(self, g):
        upconved = self.main(g)
        assert upconved.shape[-1] == self.up_size, f"Upsized image size must be equal to upsizing factor : {upconved.shape[-1]}, {self.up_size}"

        if self.up_size == self.im_size:
            return upconved
        else:
            return F.interpolate(upconved, size = self.im_size, mode='bilinear', align_corners=False)
       

def flow_warp(x, flow, interp_mode="bilinear", padding_mode="zeros"):
    """
    Warp an image or feature map with optical flow

    Copyed with modifications, from https://github.com/xinntao/BasicSR

    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2

    # scale grid to [-1,1]
    vgrid_x = 2.0 * grid[:, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * grid[:, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=2)
    vgrid_scaled.requires_grad = False
    vgrid_scaled = vgrid_scaled.type_as(x)
    vgrid = vgrid_scaled + flow
    output = F.grid_sample(x, vgrid, mode=interp_mode, padding_mode=padding_mode)

    return output


class VggEncoder(nn.Module):
    def __init__(self, im_size, nz):
        # vgg encoder
        super().__init__()
        self.im_size = im_size
        self.nz = nz
        # pretrained vgg19
        self.vgg = torchvision.models.vgg19(pretrained=True).features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, nz),
            nn.LayerNorm(nz),
        )

    def forward(self, x):
        # vgg encoder
        x = self.vgg(x)
        x = self.head(x)
        return x


class AMWActionModule(nn.Module):
    """
    Add, multiply, and Warp Action.
    """

    def __init__(self, ngf: int, nz: int):
        super().__init__()
        self.nz = nz
        self.flow_generator = SubDecoder(nz, ngf, 2)
        self.soft_bias = SubDecoder(nz, ngf, 3)
        self.soft_scale = SubDecoder(nz, ngf, 3)

        self.head = nn.Tanh()

    def forward(self, img, g):
        g = g.reshape(-1, self.nz, 1, 1)

        flow = self.flow_generator(g).permute(0, 2, 3, 1)
        bias = self.soft_bias(g)
        scale = self.soft_scale(g)
        scaled_img = self.head(img * scale + bias)

        return flow_warp(scaled_img, flow)


class UnetActionModule(nn.Module):
    def __init__(self, ngf: int, nz: int, im_size: int = 128):
        super(UnetActionModule, self).__init__()

        self.latent_unfolder = SubDecoder(nz, ngf, 16, im_size)
        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels = 16 + 3,
            classes=3,
        )
        self.nz = nz
        self.im_size = im_size

    def forward(self, img, g):
        g = g.reshape(-1, self.nz, 1, 1)

        if img.shape[-1] != self.im_size:
            img = F.interpolate(img, size=self.im_size, mode='bilinear', align_corners=False)

        merged = torch.cat((img, self.latent_unfolder(g)), 1)
        # print(merged.shape)
        x = self.unet(merged)
        # print(x.shape)
        return x


ACTIONNETWORKMAPS = {
    "amw": AMWActionModule,
    "unet": UnetActionModule,
}


class PairAction(nn.Module):
    def __init__(
        self, ngf: int, nz: int, im_size: int = 64, action_type: str = "amw",
    ):
        super().__init__()
        self.nz = nz
        self.ngf = ngf
        self.action = ACTIONNETWORKMAPS[action_type](ngf, nz, im_size)
        self.encoder = VggEncoder(im_size, nz)
        self.criterion = nn.MSELoss()
        self.im_size = im_size

    def forward(self, img1, img2):

        img1h, img2h = self.recon(img1, img2, t=1.0)

        if img1h.shape[-1] != self.im_size:
            img1h = F.interpolate(img1h, size=self.im_size, mode='bilinear', align_corners=False)
            img2h = F.interpolate(img2h, size=self.im_size, mode='bilinear', align_corners=False)


        loss1 = self.criterion(img1h, img1)
        loss2 = self.criterion(img2h, img2)

        return loss1 + loss2

    def recon(self, img1, img2, t=0.1):
        z1 = self.encoder(img1)
        z2 = self.encoder(img2)

        diff = (z1 - z2) * t

        img1h = self.action(img2, diff)  # should be img1
        img2h = self.action(img1, -diff)  # should be img2

        return img1h, img2h


class CommutativeAction(nn.Module):
    pass  # TODO : implement triplet-commutative action


class AbelianAction(nn.Module):
    pass  # TODO : implement triplet-commutative action + abelinian action

