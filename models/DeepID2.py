import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.autograd import Variable

class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size ** 2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        #print("unfold H:", x.unfold(2, kh, dh))
        # print(x.size(), x)

        x = x.unfold(3, kw, dw).unfold(2, kh, dh)
        #print("unfold H, W:", x)
        #print(x)
        #print("x.size()[:-2]:", x.size()[:-2])
        x = x.contiguous().view(*x.size()[:-2], -1)
        #print("x.unsqueeze:", x.unsqueeze(1).size())
        #print("weight:", self.weight.sum([2, -1]).size()
        # Sum in in_channel and kernel_size dims
        #print("x.size:", x.size())
        #print("self.weight:", x.unsqueeze(1)*self.weight.size())
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out



class DeepID2(nn.Module):

    def __init__(self, dim):
        super(DeepID2,self).__init__()
        inplace = True
        self.out_dim = dim
        self.conv1 = nn.Conv2d(3, 20, kernel_size=(4, 4))
        self.conv1_relu = nn.ReLU(inplace)
        self.conv1_pool = nn.MaxPool2d(kernel_size=(2,2))
        self.conv2 = nn.Conv2d(20, 40, kernel_size=(3,3))
        self.conv2_relu = nn.ReLU(inplace)
        self.conv2_pool = nn.MaxPool2d(kernel_size=(2,2))

        self.conv3 = LocallyConnected2d(40, 60, [10, 8], 3, 1, True)
        self.conv3_relu = nn.ReLU(inplace)
        self.conv3_pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv4 = LocallyConnected2d(60, 80, [4,3], 2, 1, True)
        self.conv4_relu = nn.ReLU(inplace)
        #self.fc_conv3_conv4 = nn.Linear(in_features=in_dim, out_features=out_dim)


    def forward(self, x):
        output_conv1 = self.conv1(x)
        output_conv1_relu = self.conv1_relu(output_conv1)
        output_conv1_pool = self.conv1_pool(output_conv1_relu)
        #print(output_conv1_pool.size())
        output_conv2 = self.conv2(output_conv1_pool)
        output_conv2_relu = self.conv2_relu(output_conv2)
        output_conv2_pool = self.conv2_pool(output_conv2_relu)
        #print(output_conv2_pool.size())
        output_conv3 = self.conv3(output_conv2_pool)
        output_conv3_relu = self.conv3_relu(output_conv3)
        output_conv3_pool = self.conv3_pool(output_conv3_relu)
        #print(output_conv3_pool.size())
        output_conv4 = self.conv4(output_conv3_pool)
        output_conv4_relu = self.conv4_relu(output_conv4)
        #print( output_conv4_relu.size())
        in_dim = output_conv3_pool.size()[1]*output_conv3_pool.size()[2]*output_conv3_pool.size()[3]\
            + output_conv4_relu.size()[1] * output_conv4_relu.size()[2] *output_conv4_relu.size()[3]
        out_dim = self.out_dim


        #print("in_dim:", in_dim, "out_dim:", out_dim)
        fc_input = torch.cat([output_conv4.view(output_conv4.size(0), -1),output_conv3_pool.view(output_conv3_pool.size(0), -1)], 1)
        #print(output_conv4.view(output_conv4.size(0), -1).size(), output_conv3_pool.view(output_conv3_pool.size(0), -1).size())
        fc = nn.Linear(in_dim, out_dim)
        fc_output = fc(fc_input)
        #print(fc_input.size())
        #print(fc_output.size())


        return fc_output


def main():
    model = DeepID2(160)
    images = Variable(torch.ones(8, 3, 55, 47))
    model(images)
if __name__ == '__main__':
    main()
