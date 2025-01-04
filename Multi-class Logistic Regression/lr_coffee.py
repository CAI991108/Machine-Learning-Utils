import os
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(666)


def load_data_img(path, classes, img_size=32):
    """ load image dataset
    Input:
        path: [str] path of dir which contains the subfolders of different classes' images
        classes: [list] class names
    Return:
        X: (n, d), data matrix
        Y: (n, ), corresponding label
    """
    if os.path.exists(path + 'X.npy'):
        X = np.load(path + 'X.npy')
        Y = np.load(path + 'Y.npy')
        return X, Y

    X, Y = [], []
    for y, cls in enumerate(classes):
        data_path = Path(path + cls)
        for p in data_path.iterdir():
            img = ImageOps.grayscale(Image.open(f"{p}"))
            img = img.resize((img_size, img_size))
            x = np.array(img).flatten()
            X.append(x)
            Y.append(y)

    X, Y = np.array(X), np.array(Y)
    np.save(path + 'X.npy', X)
    np.save(path + 'Y.npy', Y)

    return X, Y


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cal_loss(pred, Y_onehot):
    """ calculate multinomial logistic regression loss
    Input:
        pred: (n, K), softmax output
        Y_onehot: (n, K), onehot label
    Output:
        logistic loss
    """
    return - np.mean(np.log(np.sum(np.multiply(pred, Y_onehot), axis=1)))


def plot_losses_and_accuracies(train_losses_gd, test_losses_gd, train_accuracies_gd, test_accuracies_gd,
                               train_losses_agd, test_losses_agd, train_accuracies_agd, test_accuracies_agd,
                               plt_title=None):
    # Plot losses
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_gd, label='GD Train Loss')
    plt.plot(test_losses_gd, label='GD Test Loss')
    plt.plot(train_losses_agd, label='AGD Train Loss')
    plt.plot(test_losses_agd, label='AGD Test Loss')
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if plt_title:
        plt.title("Losses for " + plt_title)

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies_gd, label='GD Train Accuracy')
    plt.plot(test_accuracies_gd, label='GD Test Accuracy')
    plt.plot(train_accuracies_agd, label='AGD Train Accuracy')
    plt.plot(test_accuracies_agd, label='AGD Test Accuracy')
    plt.title('Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    if plt_title:
        plt.title("Accuracies for " + plt_title)
    plt.show()


# ============ code for loading coffee dataset ============
classes = ['Dark', 'Green', 'Light', 'Medium']
X, Y = load_data_img('coffee/train/', classes)
X_test, Y_test = load_data_img('coffee/test/', classes)
n, n_test = X.shape[0], X_test.shape[0]

# normalize data
mean, std = X.mean(axis=0), X.std(axis=0)
X = (X - mean) / std
X_test = (X_test - mean) / std

inlcude_bias = True
optimizers = ['gd', 'agd']

if inlcude_bias:
    X = np.concatenate([X, np.ones(shape=(n, 1))], axis=1)
    X_test = np.concatenate([X_test, np.ones(shape=(n_test, 1))], axis=1)

d = X.shape[1]
K = np.max(Y) + 1
Y_onehot = np.eye(K)[Y]  # (n, K)
Y_test_onehot = np.eye(K)[Y_test]  # (n', K)

# hyperparameters
mu = 5e-2
epochs = 1000

# initialization
Theta = np.zeros(shape=(d, K))

# train
for opt in optimizers:
    Theta = np.zeros(shape=(d, K), dtype=np.float64)
    w = np.zeros(shape=(d, K), dtype=np.float64)
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(epochs):

        # evaluate
        if epoch % 5 == 0:
            pred_train = softmax(np.matmul(X, Theta))  # (n, K)
            train_loss = cal_loss(pred_train, Y_onehot)
            train_losses.append(train_loss)
            pred_test = softmax(np.matmul(X_test, Theta))  # (n', K)
            test_loss = cal_loss(pred_test, Y_test_onehot)
            test_losses.append(test_loss)

            test_acc = np.sum(pred_test.argmax(axis=1) == Y_test) / n_test
            test_accuracies.append(test_acc)
            train_acc = np.sum(pred_train.argmax(axis=1) == Y) / n
            train_accuracies.append(train_acc)

            print(
                f"epoch:{epoch}, train_loss:{train_loss:.5f}, test_loss:{test_loss:.5f},"
                f" test_acc:{test_acc:.4f}, train_acc:{train_acc:.4f}")

        # Accelerated gradient descent
        if opt == 'agd':
            temp = Theta.copy()
            Theta = w - (1 / n) * mu * np.matmul(X.T, (-Y_onehot + softmax(np.matmul(X, w))))
            w = Theta + (epoch / (epoch + 3)) * (Theta - temp)
            train_loss_agd, test_loss_agd, train_acc_agd, test_acc_agd = (
                train_losses, test_losses, train_accuracies, test_accuracies)

        # Gradient descent
        elif opt == 'gd':
            Theta += -1 * (1 / n) * mu * np.matmul(X.T, (-Y_onehot + softmax(np.matmul(X, Theta))))
            train_loss_gd, test_loss_gd, train_acc_gd, test_acc_gd = (
                train_losses, test_losses, train_accuracies, test_accuracies)

# Plot the results
plot_losses_and_accuracies(train_loss_gd, test_loss_gd, train_acc_gd, test_acc_gd,
                           train_loss_agd, test_loss_agd, train_acc_agd, test_acc_agd, plt_title='Coffee Dataset')
