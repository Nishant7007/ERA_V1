from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F

class S7Model1(nn.Module):
    def __init__(self):
        # super is used to initialize the parent class (nn.Module) so that we can use the
        # methods and members of the parent class in Model1 class
        super(Model1, self).__init__()
        self.kernel_size = (3, 3)



        # n_out = (floor(n_in+2p-k)/s)/+1


        #Inout block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=self.kernel_size, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # k=3, s=1, p=0, n_in=32, n_out=30  j_in=1, j_out=1, r_in=1, r_out=3

        #CONVBLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        ) # k=3, s=1, p=0, n_in=30, n_out=28  j_in=1, j_out=1, r_in=3, r_out=5
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=self.kernel_size, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        ) # k=3, s=1, p=0, n_in=28, n_out=26  j_in=1, j_out=1, r_in=5, r_out=7

        #TRANSITIONBLOCK 1
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=self.kernel_size, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256)
            ) # k=3, s=1, p=0, n_in=26, n_out=24  j_in=1, j_out=1, r_in=7, r_out=9

        self.pool1 = nn.MaxPool2d(2, 2) # k=2, s=2, p=0, n_in=24, n_out=12  j_in=1, j_out=2, r_in=9, r_out=r_in+(k-1)*j_in=10


        #CONVBLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=self.kernel_size, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )# k=3, s=1, p=0, n_in=12, n_out=10  j_in=2, j_out=2, r_in=10, r_out=r_in+(k-1)*j_in=14
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=self.kernel_size, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )# k=3, s=1, p=0, n_in=10, n_out=8  j_in=2, j_out=2, r_in=14, r_out=r_in+(k-1)*j_in=18
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=self.kernel_size, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )# k=3, s=1, p=0, n_in=8, n_out=6  j_in=2, j_out=2, r_in=18, r_out=r_in+(k-1)*j_in=22

        #GAP LAYER
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
            ) # k=4, s=4, p=0, n_in=6, n_out=1  j_in=2, j_out=8, r_in=22, r_out=r_in+(k-1)*j_in=28

        #OUTPUT BLOCK
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
            )# k=1, s=1, p=0, n_in=1, n_out=1  j_in=8, j_out=8, r_in=28, r_out=r_in+(k-1)*j_in=28

    def forward(self, x):
        # print(f'input shape: {x.shape}')
        x = self.convblock1(x)
        # print(f'x after conv1: {x.shape}')
        x = self.convblock2(x)
        # print(f'x after conv2: {x.shape}')
        x = self.convblock3(x)
        # print(f'x after conv3: {x.shape}')
        x = self.convblock4(x)
        # print(f'x after conv4: {x.shape}')
        x = self.pool1(x)
        # print(f'x after pool1: {x.shape}')
        x = self.convblock5(x)
        # print(f'x after conv5: {x.shape}')
        x = self.convblock6(x)
        # print(f'x after conv6: {x.shape}')
        x = self.convblock7(x)
        # print(f'x after conv7: {x.shape}')
        x = self.gap(x)
        # print(f'x after GAP: {x.shape}')
        x = self.convblock8(x)
        # print(f'x after conv8: {x.shape}')
        x = x.view(-1, 10)
        # print(f'x after view: {x.shape}')
        return F.log_softmax(x, dim=1)
      
      

class S7Model2(nn.Module):
    def __init__(self):
        # super is used to initialize the parent class (nn.Module) so that we can use the
        # methods and members of the parent class in Model1 class
        super(Model2, self).__init__()
        self.kernel_size = (3, 3)



        # n_out = (floor(n_in+2p-k)/s)/+1


        #Inout block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=self.kernel_size, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # k=3, s=1, p=0, n_in=32, n_out=30  j_in=1, j_out=1, r_in=1, r_out=3

        #CONVBLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=self.kernel_size, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # k=3, s=1, p=0, n_in=30, n_out=28  j_in=1, j_out=1, r_in=3, r_out=5
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=self.kernel_size, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12)
        ) # k=3, s=1, p=0, n_in=28, n_out=26  j_in=1, j_out=1, r_in=5, r_out=7

        #TRANSITIONBLOCK 1
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=self.kernel_size, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
            ) # k=3, s=1, p=0, n_in=26, n_out=24  j_in=1, j_out=1, r_in=7, r_out=9

        self.pool1 = nn.MaxPool2d(2, 2) # k=2, s=2, p=0, n_in=24, n_out=12  j_in=1, j_out=2, r_in=9, r_out=r_in+(k-1)*j_in=10


        #CONVBLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=12, kernel_size=self.kernel_size, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12)
        )# k=3, s=1, p=0, n_in=12, n_out=10  j_in=2, j_out=2, r_in=10, r_out=r_in+(k-1)*j_in=14
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=self.kernel_size, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )# k=3, s=1, p=0, n_in=10, n_out=8  j_in=2, j_out=2, r_in=14, r_out=r_in+(k-1)*j_in=18
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=self.kernel_size, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )# k=3, s=1, p=0, n_in=8, n_out=6  j_in=2, j_out=2, r_in=18, r_out=r_in+(k-1)*j_in=22

        #GAP LAYER
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
            ) # k=4, s=4, p=0, n_in=6, n_out=1  j_in=2, j_out=8, r_in=22, r_out=r_in+(k-1)*j_in=28

        #OUTPUT BLOCK
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
            )# k=1, s=1, p=0, n_in=1, n_out=1  j_in=8, j_out=8, r_in=28, r_out=r_in+(k-1)*j_in=28

    def forward(self, x):
        # print(f'input shape: {x.shape}')
        x = self.convblock1(x)
        # print(f'x after conv1: {x.shape}')
        x = self.convblock2(x)
        # print(f'x after conv2: {x.shape}')
        x = self.convblock3(x)
        # print(f'x after conv3: {x.shape}')
        x = self.convblock4(x)
        # print(f'x after conv4: {x.shape}')
        x = self.pool1(x)
        # print(f'x after pool1: {x.shape}')
        x = self.convblock5(x)
        # print(f'x after conv5: {x.shape}')
        x = self.convblock6(x)
        # print(f'x after conv6: {x.shape}')
        x = self.convblock7(x)
        # print(f'x after conv7: {x.shape}')
        x = self.gap(x)
        # print(f'x after GAP: {x.shape}')
        x = self.convblock8(x)
        # print(f'x after conv8: {x.shape}')
        x = x.view(-1, 10)
        # print(f'x after view: {x.shape}')
        return F.log_softmax(x, dim=1)
      
             

class S7Model3(nn.Module):
    def __init__(self):
        # super is used to initialize the parent class (nn.Module) so that we can use the
        # methods and members of the parent class in Model1 class
        super(Model3, self).__init__()
        self.kernel_size = (3, 3)
        self.dropout_value = 0.05

        # n_out = (floor(n_in+2p-k)/s)/+1


        #Inout block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=self.kernel_size, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
          	nn.Dropout(self.dropout_value)
        ) # k=3, s=1, p=0, n_in=32, n_out=30  j_in=1, j_out=1, r_in=1, r_out=3

        #CONVBLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=self.kernel_size, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
          	nn.Dropout(self.dropout_value)
        ) # k=3, s=1, p=0, n_in=30, n_out=28  j_in=1, j_out=1, r_in=3, r_out=5
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=self.kernel_size, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
          	nn.Dropout(self.dropout_value)
        ) # k=3, s=1, p=0, n_in=28, n_out=26  j_in=1, j_out=1, r_in=5, r_out=7

        #TRANSITIONBLOCK 1
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=self.kernel_size, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
          	nn.Dropout(self.dropout_value)
            ) # k=3, s=1, p=0, n_in=26, n_out=24  j_in=1, j_out=1, r_in=7, r_out=9

        self.pool1 = nn.MaxPool2d(2, 2) # k=2, s=2, p=0, n_in=24, n_out=12  j_in=1, j_out=2, r_in=9, r_out=r_in+(k-1)*j_in=10


        #CONVBLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=12, kernel_size=self.kernel_size, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
          	nn.Dropout(self.dropout_value)
        )# k=3, s=1, p=0, n_in=12, n_out=10  j_in=2, j_out=2, r_in=10, r_out=r_in+(k-1)*j_in=14
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=self.kernel_size, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
          	nn.Dropout(self.dropout_value)
        )# k=3, s=1, p=0, n_in=10, n_out=8  j_in=2, j_out=2, r_in=14, r_out=r_in+(k-1)*j_in=18
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=self.kernel_size, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
          	nn.Dropout(self.dropout_value)
        )# k=3, s=1, p=0, n_in=8, n_out=6  j_in=2, j_out=2, r_in=18, r_out=r_in+(k-1)*j_in=22

        #GAP LAYER
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
            ) # k=4, s=4, p=0, n_in=6, n_out=1  j_in=2, j_out=8, r_in=22, r_out=r_in+(k-1)*j_in=28

        #OUTPUT BLOCK
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
            )# k=1, s=1, p=0, n_in=1, n_out=1  j_in=8, j_out=8, r_in=28, r_out=r_in+(k-1)*j_in=28

    def forward(self, x):
        # print(f'input shape: {x.shape}')
        x = self.convblock1(x)
        # print(f'x after conv1: {x.shape}')
        x = self.convblock2(x)
        # print(f'x after conv2: {x.shape}')
        x = self.convblock3(x)
        # print(f'x after conv3: {x.shape}')
        x = self.convblock4(x)
        # print(f'x after conv4: {x.shape}')
        x = self.pool1(x)
        # print(f'x after pool1: {x.shape}')
        x = self.convblock5(x)
        # print(f'x after conv5: {x.shape}')
        x = self.convblock6(x)
        # print(f'x after conv6: {x.shape}')
        x = self.convblock7(x)
        # print(f'x after conv7: {x.shape}')
        x = self.gap(x)
        # print(f'x after GAP: {x.shape}')
        x = self.convblock8(x)
        # print(f'x after conv8: {x.shape}')
        x = x.view(-1, 10)
        # print(f'x after view: {x.shape}')
        return F.log_softmax(x, dim=1)

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
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
          	nn.Dropout(self.dropout_value)
        )

        #CONVBLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
          	nn.Dropout(self.dropout_value)
        )
        #TRANSITIONBLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, padding=0, bias=False),
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
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
          	nn.Dropout(self.dropout_value)
        )
        self.pool2 = nn.MaxPool2d(2, 2)


        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
          	nn.Dropout(self.dropout_value)
        )
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
          	nn.Dropout(self.dropout_value)
        )
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
          	nn.Dropout(self.dropout_value)
        )

        #GAP LAYER
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
            )

        #OUTPUT BLOCK
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=1, bias=False)
            )

        self.linear = nn.Linear(160,10)

    def forward(self, x):
        # print(f'input shape: {x.shape}')
        x = self.convblock1(x)
        # print(f'conv1 shape: {x.shape}')
        x = x+self.convblock2(x)
        # print(f'conv2 shape: {x.shape}')
        x = self.convblock3(x)
        # print(f'conv3 shape: {x.shape}')
        x = self.pool1(x)
        # print(f'pool1 shape: {x.shape}')
        x = self.convblock4(x)
        # print(f'conv4 shape: {x.shape}')
        x = self.convblock5(x)
        # print(f'conv5 shape: {x.shape}')
        x = x + self.convblock6(x)
        # print(f'conv6 shape: {x.shape}')
        x = self.convblock7(x)
        # print(f'conv7 shape: {x.shape}')
        x = self.pool2(x)
        # print(f'pool2 shape: {x.shape}')
        x = self.convblock8(x)
        # print(f'conv8 shape: {x.shape}')
        x = x+self.convblock9(x)
        # print(f'conv9 shape: {x.shape}')
        x = self.convblock10(x)
        # print(f'conv10 shape: {x.shape}')
        x = self.gap(x)
        # print(f'gap shape: {x.shape}')
        x = self.convblock11(x)
        # print(f'conv11 shape: {x.shape}')
        # x = x.view(-1, 160)
        x = x.view(x.size(0), -1)
        # print(f'view shape: {x.shape}')
        x = self.linear(x)
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
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
          	nn.Dropout(self.dropout_value)
        )

        #CONVBLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
          	nn.Dropout(self.dropout_value)
        )
        #TRANSITIONBLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 32),
          	nn.Dropout(self.dropout_value)
        )
        self.pool1 = nn.MaxPool2d(2, 2)


        #CONVBLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 32),
          	nn.Dropout(self.dropout_value)
        )
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 32),
          	nn.Dropout(self.dropout_value)
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 32),
          	nn.Dropout(self.dropout_value)
        )
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
          	nn.Dropout(self.dropout_value)
        )
        self.pool2 = nn.MaxPool2d(2, 2)


        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
          	nn.Dropout(self.dropout_value)
        )
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
          	nn.Dropout(self.dropout_value)
        )
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 32),
          	nn.Dropout(self.dropout_value)
        )

        #GAP LAYER
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
            )

        #OUTPUT BLOCK
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=1, bias=False)
            )

        self.linear = nn.Linear(160,10)

    def forward(self, x):
        x = self.convblock1(x)
        x = x+self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = x + self.convblock6(x)
        x = self.convblock7(x)
        x = self.pool2(x)
        x = self.convblock8(x)
        x = x+self.convblock9(x)
        x = self.convblock10(x)
        x = self.gap(x)
        x = self.convblock11(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
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
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            LayerNorm(),
          	nn.Dropout(self.dropout_value)
        )

        #CONVBLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            LayerNorm(),
          	nn.Dropout(self.dropout_value)
        )
        #TRANSITIONBLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            LayerNorm(),
          	nn.Dropout(self.dropout_value)
        )
        self.pool1 = nn.MaxPool2d(2, 2)


        #CONVBLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            LayerNorm(),
          	nn.Dropout(self.dropout_value)
        )
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            LayerNorm(),
          	nn.Dropout(self.dropout_value)
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            LayerNorm(),
          	nn.Dropout(self.dropout_value)
        )
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            LayerNorm(),
          	nn.Dropout(self.dropout_value)
        )
        self.pool2 = nn.MaxPool2d(2, 2)


        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            LayerNorm(),
          	nn.Dropout(self.dropout_value)
        )
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.kernel_size, padding=1, bias=False),
            nn.ReLU(),
            LayerNorm(),
          	nn.Dropout(self.dropout_value)
        )
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, padding=1, bias=False),
            nn.ReLU(),
            LayerNorm(),
          	nn.Dropout(self.dropout_value)
        )

        #GAP LAYER
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
            )

        #OUTPUT BLOCK
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=1, bias=False)
            )

        self.linear = nn.Linear(160,10)

    def forward(self, x):
        x = self.convblock1(x)
        x = x+self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = x + self.convblock6(x)
        x = self.convblock7(x)
        x = self.pool2(x)
        x = self.convblock8(x)
        x = x+self.convblock9(x)
        x = self.convblock10(x)
        x = self.gap(x)
        x = self.convblock11(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)