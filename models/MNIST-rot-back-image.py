import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.autograd import Variable


class MinistNN(nn.Module):

    def __init__(self):
        super(MinistNN,self).__init__()
        inplace = True

        self.conv1 = nn.Conv2d(3, 96, kernel_size=(3, 3))
        self.conv1_relu = nn.ReLU(inplace)
        self.drop_out_1 = nn.Dropout(p=0.75)
        self.conv1_pool = nn.MaxPool2d(kernel_size=(3,3), stride=2)
        self.conv2 = nn.Conv2d(96, 128, kernel_size=(3, 3))
        self.conv2_relu = nn.ReLU(inplace)
        self.drop_out_2 = nn.Dropout(p=0.75)
        self.conv2_pool = nn.MaxPool2d(kernel_size=(3,3),stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3,3))
        self.conv3_relu = nn.ReLU(inplace)
        self.drop_out_3 = nn.Dropout(p=0.5)
        self.conv3_pool = nn.MaxPool2d(kernel_size=(3, 3))

        self.fc_1 = nn.Linear(24, 24)
        self.fc_2 = nn.Linear(24, 24)
        #self.conv4 = LocallyConnected2d(60, 80, [4,3], 2, 1, True)


        #self.fc_conv3_conv4 = nn.Linear(in_features=in_dim, out_features=out_dim)


    def forward(self, x):

        output_conv1 = self.conv1(x)
        output_conv1_relu = self.conv1_relu(output_conv1)
        output_conv1_dropout = self.drop_out_1(output_conv1_relu)
        output_conv1_pool = self.conv1_pool(output_conv1_dropout)

        output_conv2 = self.conv2(output_conv1_pool)
        output_conv2_relu = self.conv2_relu(output_conv2)
        output_conv2_dropout = self.drop_out_2(output_conv2_relu)
        output_conv2_pool = self.conv2_pool(output_conv2_dropout)

        #print("output_conv2_pool size, ", output_conv2_pool.size())

        output_conv3 = self.conv3(output_conv2_pool)
        output_conv3_relu = self.conv3_relu(output_conv3)
        output_conv3_dropout = self.drop_out_2(output_conv3_relu)
        output_conv3_pool = self.conv3_pool(output_conv3_dropout)


        print("output_conv3_pool size, ", output_conv3_pool.size())
        #print(output_conv3_pool.size())
        output_conv4 = self.conv4(output_conv3_pool)






def main():
    model = MinistNN()
    images = Variable(torch.ones(8, 3, 28, 28))
    model(images)
if __name__ == '__main__':
    main()
