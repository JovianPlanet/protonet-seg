import matplotlib.pyplot as plt
import torch

def plot_loss(data, n, title, ylabel, loc):
    for i in range(n):
        plt.plot(range(data.shape[1]), data[i], label=str(n-i))#+' cabezas')

    plt.legend(title="Número de cabezas", loc=loc, fontsize='small', fancybox=True, ncol=2)

    plt.xlabel('Epocas')
    plt.ylabel(ylabel)

    plt.title(title)
    plt.show()

def plot_scatter(data, n, title, ylabel, loc):
    print(data.shape[0])
    #for i in range(n):
    plt.scatter(range(data.shape[0]), data)#, label=str(i+1))#+' cabezas')

    plt.legend(title="Número de cabezas", loc=loc, fontsize='small', fancybox=True, ncol=2)

    plt.xlabel('Epocas')
    plt.ylabel(ylabel)

    plt.title(title)
    plt.show() 

NUM_HEADS = 8

PATH_FS_LOSS = './models/fsmul_LOSS-cross-ep25.pth'
PATH_FS_DICE = './models/fsmul_DICE-cross-ep25.pth'
PATH_UNET_LOSS = './models/unet_LOSS-ep25-4.pth'
PATH_UNET_DICE = './models/unet_DICE-ep25-4.pth'
PATH_UNET_TEST = './models/unetm_TEST-ep25-4.pth'

fs_loss = torch.load(PATH_FS_LOSS)
plot_loss(fs_loss, NUM_HEADS, 'fs loss (Entrenamiento)', 'Cross entropy loss', 1)

fs_dice = torch.load(PATH_FS_DICE)
plot_loss(fs_dice, NUM_HEADS, 'fs Dice (Validación)', 'Dice coefficient', 4)

unet_loss = torch.load(PATH_UNET_LOSS)
plot_loss(unet_loss, NUM_HEADS, 'Unet loss (Entrenamiento)', 'Cross entropy loss', 1)

unet_dice = torch.load(PATH_UNET_DICE)
plot_loss(unet_dice, NUM_HEADS, 'Unet Dice (Validación)', 'Dice coefficient', 4)

unet_test = torch.load(PATH_UNET_TEST)
plot_scatter(unet_test, NUM_HEADS, 'Unet Dice (Test)', 'Dice coefficient', 4)