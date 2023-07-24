from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F


class Custom_Resnet(nn.Module):
    def __init__(self):
        super(Custom_Resnet, self).__init__()
        self.kernel_size = (3, 3)
        self.dropout_value = 0.05

        #PrepLayer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=self.kernel_size,
                                             stride=1, padding=1, bias=False),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(64),
                                   nn.Dropout(self.dropout_value))

        #Layer1
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=self.kernel_size,
                                             stride=1, padding=1, bias=False),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(128),
                                   nn.Dropout(self.dropout_value))

        self.res1 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=self.kernel_size,
                                             stride=1, padding=1, bias=False),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(128),
                                  nn.Dropout(self.dropout_value),
                                  nn.Conv2d(in_channels=128, out_channels=128, kernel_size=self.kernel_size,
                                             stride=1, padding=1, bias=False),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(128),
                                  nn.Dropout(self.dropout_value))


        #Layer2
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=self.kernel_size, padding=1),
                            nn.MaxPool2d(2,2),
                            nn.ReLU(),
                            nn.BatchNorm2d(256),
                            nn.Dropout(self.dropout_value))

        #Layer3
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=self.kernel_size,
                                             stride=1, padding=1, bias=False),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(512),
                                   nn.Dropout(self.dropout_value))

        self.res2 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=self.kernel_size,
                                             stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(512),
                                   nn.Dropout(self.dropout_value),
                                   nn.Conv2d(in_channels=512, out_channels=512, kernel_size=self.kernel_size,
                                             stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(512),
                                   nn.Dropout(self.dropout_value))

        self.pool1 = nn.MaxPool2d(kernel_size = 4, stride =2)
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        #PrerLayer
        x = self.conv1(x)
        #Layer1
        x = self.conv2(x)
        r1 = self.res1(x)
        x = x + r1
        #Layer2
        x = self.conv3(x)
        #Layer3
        x = self.conv4(x)
        r2 = self.res2(x)
        x = x + r2
        x = self.pool1(x)
        out = x.view(x.size(0),-1)
        x = self.fc1(out)
        return F.log_softmax(x, dim=-1)
