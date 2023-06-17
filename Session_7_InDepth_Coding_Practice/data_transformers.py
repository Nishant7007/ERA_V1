from torchvision import datasets, transforms

class Transforms:
    def __init__(self):
        self.train_transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

        self.test_transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
        ])

    def download_train_data(self):
        train = datasets.MNIST('./data', train=True, download=True, transform=self.train_transforms)
        return train

    def download_test_data(self):
        test = datasets.MNIST('./data', train=False, download=False, transform=self.test_transforms)
        return test
      
      
class TransformsWithAugmentation:
    def __init__(self):
        self.train_transforms = transforms.Compose([
          						transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

        self.test_transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
        ])

    def download_train_data(self):
        train = datasets.MNIST('./data', train=True, download=True, transform=self.train_transforms)
        return train

    def download_test_data(self):
        test = datasets.MNIST('./data', train=False, download=False, transform=self.test_transforms)
        return test