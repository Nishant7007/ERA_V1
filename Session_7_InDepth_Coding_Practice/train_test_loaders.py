from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

SEED = 1

class Train_Test_Dataloaders:

    #check if CUDA is available
    def check_if_cuda_available(self):
        cuda = torch.cuda.is_available()
        print("is CUDA available? ", cuda)
        if(cuda):
            torch.cuda.manual_seed(SEED)

    #get the arguments for dataloader
    def get_dataloader_args(self):
        is_cuda = self.check_if_cuda_available
        dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if is_cuda else dict(shuffle=True, Batch_size=64)
        return dataloader_args

    #get train dataloader
    def get_train_dataloader(self, train):
        dataloader_args = self.get_dataloader_args()
        train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
        return train_loader

    #get test dataloader
    def get_test_dataloader(self, test):
        dataloader_args = self.get_dataloader_args()
        test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
        return test_loader