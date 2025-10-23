import torch
import torch.nn as nn


def conv_block(inp_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(inp_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Unet(nn.Module):

    def __init__(self, input_ch=3, n_class=1, base_ch=64, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = conv_block(input_ch, base_ch)

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(base_ch, base_ch * 2)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(base_ch * 2, base_ch * 4)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(base_ch * 4, base_ch * 8)
        )

        self.up1 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.conv_up1 = conv_block(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.conv_up2 = conv_block(base_ch * 4, base_ch * 2)

        self.up3 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.conv_up3 = conv_block(base_ch * 2, base_ch)

        self.out = nn.Conv2d(base_ch, n_class, 1)

    def forward(self, x):
        # Encoder
        c1 = self.conv(x)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)

        # Decoder
        u1 = self.up1(c4)
        u1 = torch.cat([u1, c3], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, c2], dim=1)
        u2 = self.conv_up2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, c1], dim=1)
        u3 = self.conv_up3(u3)

        out = self.out(u3)
        return out
