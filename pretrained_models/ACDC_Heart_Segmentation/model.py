import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=4):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.downs = nn.ModuleList([
            DoubleConv(img_ch, 64),
            DoubleConv(64, 128),
            DoubleConv(128, 256),
            DoubleConv(256, 512)
        ])

        self.bottleneck = DoubleConv(512, 1024)

        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        ])
        
        self.attention_gates = nn.ModuleList([
            AttentionGate(F_g=512, F_l=512, F_int=256),
            AttentionGate(F_g=256, F_l=256, F_int=128),
            AttentionGate(F_g=128, F_l=128, F_int=64),
            AttentionGate(F_g=64, F_l=64, F_int=32)
        ])
        
        self.up_convs = nn.ModuleList([
            DoubleConv(1024, 512),
            DoubleConv(512, 256),
            DoubleConv(256, 128),
            DoubleConv(128, 64)
        ])

        self.final_conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.downs[0](x)
        e2 = self.downs[1](self.Maxpool(e1))
        e3 = self.downs[2](self.Maxpool(e2))
        e4 = self.downs[3](self.Maxpool(e3))

        b = self.bottleneck(self.Maxpool(e4))

        d4 = self.ups[0](b)
        x4 = self.attention_gates[0](g=d4, x=e4)
        d4 = self.up_convs[0](torch.cat((x4, d4), dim=1))

        d3 = self.ups[1](d4)
        x3 = self.attention_gates[1](g=d3, x=e3)
        d3 = self.up_convs[1](torch.cat((x3, d3), dim=1))

        d2 = self.ups[2](d3)
        x2 = self.attention_gates[2](g=d2, x=e2)
        d2 = self.up_convs[2](torch.cat((x2, d2), dim=1))

        d1 = self.ups[3](d2)
        x1 = self.attention_gates[3](g=d1, x=e1)
        d1 = self.up_convs[3](torch.cat((x1, d1), dim=1))

        return self.final_conv(d1)
