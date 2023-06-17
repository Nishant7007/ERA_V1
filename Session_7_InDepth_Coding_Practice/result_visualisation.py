import matplotlib.pyplot as plt


class Result_Visualisation:
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