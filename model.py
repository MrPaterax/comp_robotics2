import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniUNet(nn.Module):
    # TODO: implement a neural network as described in the handout
    def __init__(self):
        """Initialize the layers of the network as instance variables."""
        super(MiniUNet, self).__init__()
        self.left_conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size = 3, padding = 'same'), nn.ReLU(inplace = True))
        self.left_conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size = 3, padding = 'same'), nn.ReLU(inplace = True))
        self.left_conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size = 3, padding = 'same'), nn.ReLU(inplace = True))
        self.left_conv4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size = 3, padding = 'same'), nn.ReLU(inplace = True))
        self.left_conv5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size = 3, padding = 'same'), nn.ReLU(inplace = True))
        
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.right_conv1 = nn.Sequential(nn.Conv2d(384, 128, kernel_size = 3, padding = 'same'), nn.ReLU(inplace = True))
        self.right_conv2 = nn.Sequential(nn.Conv2d(192, 64, kernel_size = 3, padding = 'same'), nn.ReLU(inplace = True))
        self.right_conv3 = nn.Sequential(nn.Conv2d(96, 32, kernel_size = 3, padding = 'same'), nn.ReLU(inplace = True))
        self.right_conv4 = nn.Sequential(nn.Conv2d(48, 16, kernel_size = 3, padding = 'same'), nn.ReLU(inplace = True))
        
        self.last = nn.Conv2d(16, 6, kernel_size = 1)

    def forward(self, x):
        """
        In:
            x: Tensor [batchsize, channel, height, width], channel=3 for rgb input
        Out:
            output: Tensor [batchsize, class, height, width], class=number of objects + 1 for background
        Purpose:
            Forward process. Pass the input x through the layers defined in __init__() to get the output.
        """
        # TODO
        l1 = self.left_conv1(x)
        l2 = self.maxpool(l1)
        l3 = self.left_conv2(l2)
        l4 = self.maxpool(l3)
        l5 = self.left_conv3(l4)
        l6 = self.maxpool(l5)
        l7 = self.left_conv4(l6)
        l8 = self.maxpool(l7)
        l9 = self.left_conv5(l8)

        r1 = F.interpolate(l9, scale_factor=2)
        
        r2 = torch.cat((l7, r1), 1)
        r3 = self.right_conv1(r2)

        r4 = F.interpolate(r3, scale_factor=2)
        r5 = torch.cat((l5, r4), 1)
        r6 = self.right_conv2(r5)

        r7 = F.interpolate(r6, scale_factor=2)
        r8 = torch.cat((l3, r7), 1)
        r9 = self.right_conv3(r8)

        r10 = F.interpolate(r9, scale_factor=2)
        r11 = torch.cat((l1, r10), 1)
        r12 = self.right_conv4(r11)
        output = self.last(r12)
        return output


if __name__ == '__main__':
    model = MiniUNet()
    input_tensor = torch.zeros([1, 3, 240, 320])
    output = model(input_tensor)
    print("output size:", output.size())
    print(model)
