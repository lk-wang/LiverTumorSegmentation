
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


class DoubleConv(nn.Module):
    """
    Double Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch,out_ch,kernel_size=2,stride=2),
            # nn.Upsample(scale_factor=2),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()

        n1 = 32    # n1=32,瓶颈处为512，n1=64,瓶颈处为1024
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConv(in_ch, filters[0])      # 3-32
        self.Conv2 = DoubleConv(filters[0], filters[1]) # 32-64
        self.Conv3 = DoubleConv(filters[1], filters[2]) # 64-128
        self.Conv4 = DoubleConv(filters[2], filters[3]) # 128-256
        self.Conv5 = DoubleConv(filters[3], filters[4]) # 256-512

        self.Up5 = up_conv(filters[4], filters[3])      # 512-256
        self.Up_conv5 = DoubleConv(filters[4], filters[3]) # 512-256

        self.Up4 = up_conv(filters[3], filters[2])        # 256-128
        self.Up_conv4 = DoubleConv(filters[3], filters[2]) # 256-128

        self.Up3 = up_conv(filters[2], filters[1])         # 128-64
        self.Up_conv3 = DoubleConv(filters[2], filters[1]) # 128-64

        self.Up2 = up_conv(filters[1], filters[0])         # 64-32
        self.Up_conv2 = DoubleConv(filters[1], filters[0]) # 64-32

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)  # 32-1

        self.active = torch.nn.Sigmoid()

    def forward(self, x):   # 3

        e1 = self.Conv1(x) # 32

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2) # 64

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3) # 128

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4) # 256

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)  # 512

        d5 = self.Up5(e5)    # 256
        d5 = torch.cat((e4, d5), dim=1)   # 256+256=512
        d5 = self.Up_conv5(d5)  # 256

        d4 = self.Up4(d5)      # 128
        d4 = torch.cat((e3, d4), dim=1)   # 128+128=256
        d4 = self.Up_conv4(d4) # 128

        d3 = self.Up3(d4)     # 64
        d3 = torch.cat((e2, d3), dim=1)   # 64+64=128
        d3 = self.Up_conv3(d3)  # 64

        d2 = self.Up2(d3)     # 32
        d2 = torch.cat((e1, d2), dim=1)   # 32+32=64
        d2 = self.Up_conv2(d2)  # 32

        out = self.Conv(d2)    # 1
        out= self.active(out)
        return out
if __name__ == '__main__':
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    x = torch.rand(2, 3, 512, 512)
    #x = torch.rand(2, 3, 480, 480)
    # x = torch.rand(2, 64, 256, 256)
    model=U_Net(3,1)
    # model=up_conv(64,64)
    print("Number of model parameters:", count_params(model))  # Number of model parameters: 8,637,345
    out = model(x)
    print(out.shape)  #