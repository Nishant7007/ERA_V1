from __future__ import print_function
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR, OneCycleLR

class Train_Test_With_LR_Scheduler:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []
        self.misclassified_images = []
        self.original_labels = []
        self.predicted_labels = []

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
                indices = [item[0] for item in (pred.eq(target.view_as(pred))==False)]
                mis_img = data[indices,:,:,:]
                self.misclassified_images.extend(mis_img)
                self.original_labels.extend(target[indices])
                self.predicted_labels.extend(pred[indices])
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
        # scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
        scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=epochs)
        for epoch in range(epochs):
            print(f'EPOCH: {epoch}')
            self.train(model, device, train_loader, optimizer, epoch)
            scheduler.step()
            self.test(model, device, test_loader)
        return self.train_losses, self.test_losses, self.train_acc, self.test_acc

class ResultVisualisation:
    def plot_accuracy_and_loss(self, train_losses, test_losses, train_acc, test_acc):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[0, 0].plot([loss.cpu().detach().numpy() for loss in train_losses])
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(test_acc)
        axs[1, 1].set_title("Test Accuracy")