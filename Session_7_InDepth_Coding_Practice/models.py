import torch.nn.functional as F
import torch.nn as nn

class Model1(nn.Module):
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
      
      

class Model2(nn.Module):
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
      
             

class Model3(nn.Module):
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