import torch
import torch.nn as nn

from abelian_actions import SubDecoder, VggEncoder

class AutoEncoder(nn.Module):
    def __init__(self, im_size, nz, affine_latent, ngf, nc, ident_preserving):
        super(AutoEncoder, self).__init__()
        self.encoder = VggEncoder(
            im_size=im_size,
            nz=nz,
            affine_latent=affine_latent
        )
        self.decoder = SubDecoder(
            nz=nz,
            ngf=int(ngf*1.7),
            nc=nc,
            im_size=im_size,
            ident_preserving=ident_preserving
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    sample_image = torch.randn(1, 3, 256, 256)
    model = AutoEncoder()
    print(model(sample_image).shape)