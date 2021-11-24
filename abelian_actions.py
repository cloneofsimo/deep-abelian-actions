import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import ConvTranspose1d
import torchvision
import segmentation_models_pytorch as smp


class Upconv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super(Upconv, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.main(x)
        return x


class SubDecoder(nn.Module):
    def __init__(self, nz, ngf, nc, im_size, ident_preserving):
        super(SubDecoder, self).__init__()

        self.nz = nz
        self.im_size = im_size
        self.ident_preserving = ident_preserving
        self.upfactor = round(math.log2(im_size)) - 3
        assert self.upfactor > 0, "Image size must be greator than 8"

        self.up_size = round(2 ** (self.upfactor + 3))
        print("Decoder upsizing factor:", self.upfactor)
        print(f"img size {self.im_size}, up_size {self.up_size}")

        self.main = nn.Sequential(
            Upconv(nz, ngf * (2 ** self.upfactor), 4, 1, 0),
            *[
                Upconv(
                    ngf * (2 ** mult),
                    ngf * (2 ** (mult - 1)),
                    4,
                    2,
                    1,
                )
                for mult in range(self.upfactor, 0, -1)
            ],
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
        )

    def forward(self, g: torch.Tensor):
        upconved = self.main(g)

        if self.ident_preserving:
            g1 = g.reshape(-1, self.nz).norm(p=1, dim=1).reshape(-1, 1, 1, 1) / self.nz
            upconved = upconved * g1

        assert (
            upconved.shape[-1] == self.up_size
        ), f"Upsized image size must be equal to upsizing factor : {upconved.shape[-1]}, {self.up_size}"

        if self.up_size == self.im_size:
            return upconved
        else:
            return F.interpolate(
                upconved, size=self.im_size, mode="bilinear", align_corners=False
            )


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
    output = F.grid_sample(
        x, vgrid, mode=interp_mode, padding_mode=padding_mode, align_corners=True
    )

    return output


class VggEncoder(nn.Module):
    def __init__(self, im_size: int, nz: int, affine_latent: bool):
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
            nn.LayerNorm(nz, elementwise_affine=affine_latent),
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

    def __init__(self, ngf: int, nz: int, im_size: int, ident_preserving: bool):
        super().__init__()
        self.nz = nz
        self.flow_generator = SubDecoder(nz, ngf, 2, im_size, ident_preserving)
        self.soft_bias = SubDecoder(nz, ngf, 3, im_size, ident_preserving)
        self.soft_scale = SubDecoder(nz, ngf, 3, im_size, ident_preserving)

    def forward(self, img, g):
        g = g.reshape(-1, self.nz, 1, 1)

        flow = self.flow_generator(g).permute(0, 2, 3, 1)
        bias = self.soft_bias(g)
        scale = self.soft_scale(g)
        scaled_img = img * (1 + scale) + bias

        return flow_warp(scaled_img, flow)


class WarperActionModule(nn.Module):
    """
    Warp Action.
    """

    def __init__(self, ngf: int, nz: int, im_size: int, ident_preserving: bool):
        super().__init__()
        self.nz = nz
        self.flow_generator = SubDecoder(nz, ngf, 2, im_size, ident_preserving)

    def forward(self, img, g):
        g = g.reshape(-1, self.nz, 1, 1)

        flow = self.flow_generator(g).permute(0, 2, 3, 1)

        return flow_warp(img, flow)


class UnetActionModule(nn.Module):
    def __init__(
        self, ngf: int, nz: int, im_size: int = 128, ident_preserving: bool = False
    ):
        super(UnetActionModule, self).__init__()

        self.latent_unfolder = SubDecoder(nz, ngf, 16, im_size, ident_preserving)
        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=16 + 3,
            classes=3,
        )
        self.nz = nz
        self.im_size = im_size

    def forward(self, img, g):
        g = g.reshape(-1, self.nz, 1, 1)

        if img.shape[-1] != self.im_size:
            img = F.interpolate(
                img, size=self.im_size, mode="bilinear", align_corners=False
            )

        merged = torch.cat((img, self.latent_unfolder(g)), 1)

        x = self.unet(merged)

        return x


ACTIONNETWORKMAPS = {
    "amw": AMWActionModule,
    "unet": UnetActionModule,
    "wrp": WarperActionModule,
}


class PairAction(nn.Module):
    def __init__(
        self,
        ngf: int,
        nz: int,
        im_size: int,
        action_type: str,
        loss_type: str,
        affine_latent: bool = False,
        ident_preserving: bool = False,
        space: str = "R^n",
    ):
        super().__init__()
        self.nz = nz
        self.ngf = ngf
        self.space = space
        if self.space == "S1^n":
            self.action = ACTIONNETWORKMAPS[action_type](
                ngf, nz * 2, im_size, ident_preserving
            )
        else:
            self.action = ACTIONNETWORKMAPS[action_type](
                ngf, nz, im_size, ident_preserving
            )

        self.encoder = VggEncoder(im_size, nz, affine_latent=affine_latent)
        self.criterion = nn.MSELoss()
        self.im_size = im_size
        self.loss_type = loss_type

    def _unit_circle_exponentiate(self, z):
        zc = torch.cos(z)
        zs = torch.sin(z)

    def subtract(self, z1, z2, t=1.0):
        # calculate z1 * z2^-1

        # z must be in size B, nz

        if self.space == "R^n":
            return (z1 - z2) * t
        elif self.space == "S1^n":
            # in case of tori, we actually assume z1, z2 to be lie algebra instead.
            return torch.cat([torch.sin((z1 - z2) * t), torch.cos((z1 - z2) * t)], 1)

    def forward(self, img1, img2, img3=None):

        if self.loss_type == "simple":

            img1h, img2h = self.pair_recon(img1, img2, t=1.0)

            loss1 = self.criterion(img1h, img1)
            loss2 = self.criterion(img2h, img2)

            return loss1 + loss2

        if self.loss_type == "inverse":
            return self.inverse_loss(img1, img2)

        if self.loss_type == "compat":
            return self.compatibility_loss(img1, img2, img3)

        if self.loss_type == "abel_compat":
            return self.abelian_compatibility_loss(img1, img2, img3)

    def pair_recon(self, img1, img2, t=1.0):
        z1 = self.encoder(img1)
        z2 = self.encoder(img2)

        diff = self.subtract(z1, z2, t)
        diffi = self.subtract(z2, z1, t)

        img1h = self.action(img2, diff)  # should be img1
        img2h = self.action(img1, diffi)  # should be img2

        return img1h, img2h

    def compatibility_loss(self, iA, iB, iC):
        gA = self.encoder(iA)
        gB = self.encoder(iB)
        gC = self.encoder(iC)

        iBh = self.action(iA, self.subtract(gB, gA))  # should be iB
        iCh = self.action(iBh, self.subtract(gC, gB))  # should be iC

        return self.criterion(iCh, iC)

    def abelian_compatibility_loss(self, iA, iB, iC):

        gA = self.encoder(iA)
        gB = self.encoder(iB)
        gC = self.encoder(iC)

        iMid = self.action(iA, self.subtract(gC, gB))  # should represent gA + gC - gB
        iCh = self.action(iMid, self.subtract(gB, gA))  # should represent gC. Thus, iCh

        return self.criterion(iCh, iC)

    def inverse_loss(self, iA, iB):
        gA = self.encoder(iA)
        gB = self.encoder(iB)

        iBh = self.action(iA, self.subtract(gB, gA))  # should be iB
        iAh = self.action(iB, self.subtract(gA, gB))  # should be iA

        iAhh = self.action(iBh, self.subtract(gA, gB))  # should be iA
        iBhh = self.action(iAh, self.subtract(gB, gA))  # should be iB

        return (
            self.criterion(iAhh, iA)
            + self.criterion(iBhh, iB)
            + self.criterion(iBh, iB)
            + self.criterion(iAh, iA)
        )

    def forward_vector(self, g, base_image, t):
        # g = g.reshape(-1, self.nz, 1, 1)
        g_base = self.encoder(base_image)

        diff = self.subtract(g, g_base, t)
        return self.action(base_image, diff)
