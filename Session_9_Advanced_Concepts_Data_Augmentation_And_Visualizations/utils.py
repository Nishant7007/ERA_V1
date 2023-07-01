from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
import warnings
warnings.filterwarnings("ignore")
import random

from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import random
import albumentations as A


class TrainTransforms():
    def __init__(self):
        self.mean = (0.4914, 0.4822, 0.4471)
        self.std = [0.2469, 0.2433, 0.2615]
        self.transform = A.Compose([
            A.HorizontalFlip(),
            A.ShiftScaleRotate(),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=[0.4914*255, 0.4822*255, 0.4471*255], mask_fill_value=None, always_apply=False, p=0.3),
            A.Normalize(mean=self.mean,std=self.std),
            ToTensorV2()
        ])
    def __call__(self,img):
        img = self.transform(image=np.array(img))['image']
        return img


class TestTransforms():
    def __init__(self):
        self.mean = (0.4914, 0.4822, 0.4471)
        self.std = [0.2469, 0.2433, 0.2615]
        self.transform = A.Compose([
            A.Normalize(mean=self.mean,std=self.std),
            ToTensorV2()
        ])
    def __call__(self,img):
        img = self.transform(image=np.array(img))['image']
        return img


class DataDownloader:
    def download_train_data(self):
        train = datasets.CIFAR10('./data', train=True, download=True, transform=TrainTransforms())
        return train

    def download_test_data(self):
        test = datasets.CIFAR10('./data', train=False, download=False, transform=TestTransforms())
        return test



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
    original_labels = [i.cpu() for i in original_labels]
    predicted_labels = [i.cpu() for i in predicted_labels]

    original_labels = [original_labels[i] for i in indices]
    predicted_labels = [predicted_labels[i] for i in indices]
    images = [images[i] for i in indices]

    print(len(original_labels), len(predicted_labels), len(images))

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