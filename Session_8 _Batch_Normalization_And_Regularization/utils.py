from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
!pip install summary
from torchvision import datasets, transforms
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")


class Transforms:
    def __init__(self):
        self.train_transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                                     ])

        self.test_transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
        ])

    def download_train_data(self):
        train = datasets.CIFAR10('./data', train=True, download=True, transform=self.train_transforms)
        return train

    def download_test_data(self):
        test = datasets.CIFAR10('./data', train=False, download=False, transform=self.test_transforms)
        return test

from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import random



SEED = 1

class TrainTestDataloaders:

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


class SampleData:
  def show_sample_images(self, train_loader, classes):
    images, labels = next(iter(train_loader))
    fig = plt.figure(figsize=(10,10))
    for index in range(1, 26):
        plt.subplot(5, 5, index)
        plt.axis('off')
        data = np.clip(images[index].permute(1, 2, 0).numpy(), 0, 1)
        plt.imshow(data)
        plt.title(classes[labels[index]])
    plt.show()

  def show_misclassified_images(self, images, original_labels, predicted_labels, classes):
    indices = random.sample(range(1, len(images)), 25)
    images = [i.cpu() for i in images]
    images = [images[i] for i in indices]
    original_labels = [i.cpu() for i in original_labels]
    original_labels = [original_labels[i] for i in indices]
    predicted_labels = [i.cpu() for i in predicted_labels]
    predicted_labels = [predicted_labels[i] for i in indices]
    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(12, 15))
    axs = axs.ravel()
    for i, ax in enumerate(axs):
      ax.set_xticks([])
      ax.set_yticks([])
      data = np.clip(images[i].permute(1, 2, 0).numpy(), 0, 1)
      ax.imshow(data)
      ax.set_title(f'original: {classes[original_labels[i]]}, \n predicted: {classes[predicted_labels[i].item()]}')
    plt.tight_layout()
    plt.show()