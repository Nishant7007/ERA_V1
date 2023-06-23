from __future__ import print_function

import warnings

warnings.filterwarnings("ignore")
!pip install summary
from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


class ModelBN(nn.Module):
    def __init__(self):
        # super is used to initialize the parent class (nn.Module) so that we can use the
        # methods and members of the parent class in Model1 class
        super(ModelBN, self).__init__()
        self.kernel_size = (3, 3)
        self.dropout_value = 0.05

        # n_out = (floor(n_in+2p-k)/s)/+1


        #Inout block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
          	nn.Dropout(self.dropout_value)
        )

        #CONVBLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
          	nn.Dropout(self.dropout_value)
        )
        #TRANSITIONBLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
          	nn.Dropout(self.dropout_value)
        )
        self.pool1 = nn.MaxPool2d(2, 2)


        #CONVBLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
          	nn.Dropout(self.dropout_value)
        )
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
          	nn.Dropout(self.dropout_value)
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
          	nn.Dropout(self.dropout_value)
        )
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
          	nn.Dropout(self.dropout_value)
        )
        self.pool2 = nn.MaxPool2d(2, 2)


        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
          	nn.Dropout(self.dropout_value)
        )
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
          	nn.Dropout(self.dropout_value)
        )
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
          	nn.Dropout(self.dropout_value)
        )

        #GAP LAYER
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
            )

        #OUTPUT BLOCK
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=1, bias=False)
            )

    def forward(self, x):
        x = self.convblock1(x)
        x = x+self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = x + self.convblock5(x)
        x = x + self.convblock6(x)
        x = self.convblock7(x)
        x = self.pool2(x)
        x = x+self.convblock8(x)
        x = x+self.convblock9(x)
        x = self.convblock10(x)
        x = self.gap(x)
        x = self.convblock11(x)
        x = x.view(-1, 90)
        return F.log_softmax(x, dim=1)


class ModelGN(nn.Module):
    def __init__(self):
        # super is used to initialize the parent class (nn.Module) so that we can use the
        # methods and members of the parent class in Model1 class
        super(ModelGN, self).__init__()
        self.kernel_size = (3, 3)
        self.dropout_value = 0.05

        # n_out = (floor(n_in+2p-k)/s)/+1


        #Inout block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, 32),
          	nn.Dropout(self.dropout_value)
        )

        #CONVBLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(2, 32),
          	nn.Dropout(self.dropout_value))

        #TRANSITIONBLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(2, 32),
          	nn.Dropout(self.dropout_value))
        self.pool1 = nn.MaxPool2d(2, 2)


        #CONVBLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(2, 32),
          	nn.Dropout(self.dropout_value))
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(2, 32),
          	nn.Dropout(self.dropout_value))
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(2, 32),
          	nn.Dropout(self.dropout_value))
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(2, 32),
          	nn.Dropout(self.dropout_value))
        self.pool2 = nn.MaxPool2d(2, 2)


        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(2, 32),
          	nn.Dropout(self.dropout_value))
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(2, 32),
          	nn.Dropout(self.dropout_value))
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(2, 16),
          	nn.Dropout(self.dropout_value)
        )

        #GAP LAYER
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6))

        #OUTPUT BLOCK
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=1, bias=False))

    def forward(self, x):
        x = self.convblock1(x)
        x = x+self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = x + self.convblock5(x)
        x = x + self.convblock6(x)
        x = self.convblock7(x)
        x = self.pool2(x)
        x = x+self.convblock8(x)
        x = x+self.convblock9(x)
        x = self.convblock10(x)
        x = self.gap(x)
        x = self.convblock11(x)
        x = x.view(-1, 90)
        return F.log_softmax(x, dim=1)


class LayerNorm(nn.Module):
   def __init__(self, eps=1e-5):
       super().__init__()
       self.eps = eps

   def forward(self, x):
       N, C, H, W = x.size()
       mean = x.mean(0, keepdim=True)
       var = x.var(0, keepdim=True)
       x = (x - mean) / (var + self.eps).sqrt()
       x = x.view(N, C, H, W)
       return x


class ModelLN(nn.Module):
    def __init__(self):
        # super is used to initialize the parent class (nn.Module) so that we can use the
        # methods and members of the parent class in Model1 class
        super(ModelLN, self).__init__()
        self.kernel_size = (3, 3)
        self.dropout_value = 0.05

        # n_out = (floor(n_in+2p-k)/s)/+1


        #Inout block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            LayerNorm(),
          	nn.Dropout(self.dropout_value)
        )

        #CONVBLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            LayerNorm(),
          	nn.Dropout(self.dropout_value))

        #TRANSITIONBLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=1, bias=False),
            nn.ReLU(),
            LayerNorm(),
          	nn.Dropout(self.dropout_value))
        self.pool1 = nn.MaxPool2d(2, 2)


        #CONVBLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            LayerNorm(),
          	nn.Dropout(self.dropout_value))
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            LayerNorm(),
          	nn.Dropout(self.dropout_value))
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            LayerNorm(),
          	nn.Dropout(self.dropout_value))
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=1, bias=False),
            nn.ReLU(),
            LayerNorm(),
          	nn.Dropout(self.dropout_value))
        self.pool2 = nn.MaxPool2d(2, 2)


        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            LayerNorm(),
          	nn.Dropout(self.dropout_value))
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            LayerNorm(),
          	nn.Dropout(self.dropout_value))
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=1, bias=False),
            nn.ReLU(),
            LayerNorm(),
          	nn.Dropout(self.dropout_value)
        )

        #GAP LAYER
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6))

        #OUTPUT BLOCK
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=1, bias=False))

    def forward(self, x):
        x = self.convblock1(x)
        x = x+self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = x + self.convblock5(x)
        x = x + self.convblock6(x)
        x = self.convblock7(x)
        x = self.pool2(x)
        x = x+self.convblock8(x)
        x = x+self.convblock9(x)
        x = self.convblock10(x)
        x = self.gap(x)
        x = self.convblock11(x)
        x = x.view(-1, 90)
        return F.log_softmax(x, dim=1)