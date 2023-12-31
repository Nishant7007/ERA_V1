from __future__ import print_function
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR


class Train_Test:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

    def train(self, model, device, train_loader, optimizer, epoch):
        model.train()
        pbar = tqdm(train_loader)
        correct = 0
        processed= 0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            y_pred = model(data)

            loss = F.nll_loss(y_pred, target)
            self.train_losses.append(loss)

            loss.backward()
            optimizer.step()

            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc=f'loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

            self.train_acc.append(100*correct/processed)

    def test(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            self.test_losses.append(test_loss)

            print('\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

            self.test_acc.append(100. * correct / len(test_loader.dataset))

    def train_and_test_model(self, model, device, epochs, train_loader, test_loader):
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        for epoch in range(epochs):
            print(f'EPOCH: {epoch}')
            self.train(model, device, train_loader, optimizer, epoch)
            self.test(model, device, test_loader)
        return self.train_losses, self.test_losses, self.train_acc, self.test_acc
      
      
      
class Train_Test_With_LR_Scheduler:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

    def train(self, model, device, train_loader, optimizer, epoch):
        model.train()
        pbar = tqdm(train_loader)
        correct = 0
        processed= 0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            y_pred = model(data)

            loss = F.nll_loss(y_pred, target)
            self.train_losses.append(loss)

            loss.backward()
            optimizer.step()

            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc=f'loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

            self.train_acc.append(100*correct/processed)

    def test(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            self.test_losses.append(test_loss)

            print('\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

            self.test_acc.append(100. * correct / len(test_loader.dataset))

    def train_and_test_model(self, model, device, epochs, train_loader, test_loader):
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
        for epoch in range(epochs):
            print(f'EPOCH: {epoch}')
            self.train(model, device, train_loader, optimizer, epoch)
            scheduler.step()
            self.test(model, device, test_loader)
        return self.train_losses, self.test_losses, self.train_acc, self.test_acc