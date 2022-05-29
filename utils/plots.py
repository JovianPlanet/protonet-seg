import matplotlib.pyplot as plt

import torch

def plot_batch(x, y):

    x = x.squeeze(1)

    for im in range(x.shape[0]):

    	fig = plt.figure(figsize=(16, 16))
    	fig.subplots_adjust(hspace=1, wspace=1)

    	ax1 = fig.add_subplot(1, 2, 1)
    	ax1.axis("off")
    	ax1.imshow(x[im, :, :].cpu().detach().numpy(), cmap="gray")

    	ax2 = fig.add_subplot(1, 2, 2)
    	ax2.axis("off")
    	ax2.imshow(y[im, :, :].cpu().detach().numpy(), cmap="gray")

    	fig.tight_layout()
    	plt.show()

def plot_batch_full(x, y, p):

    x = x.squeeze(1)

    for im in range(x.shape[0]):

        fig = plt.figure(figsize=(16, 16))
        fig.subplots_adjust(hspace=1, wspace=1)

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.axis("off")
        ax1.imshow(x[im, :, :].cpu().detach().numpy(), cmap="gray")

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.axis("off")
        ax2.imshow(y[im, :, :].cpu().detach().numpy(), cmap="gray")

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.axis("off")
        ax3.imshow(p[im, :, :].cpu().detach().numpy(), cmap="gray")

        fig.tight_layout()
        plt.show()

def plot_single(x, y):
    x = x.squeeze(0)
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(hspace=1, wspace=1)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.axis("off")
    ax1.imshow(x.cpu().detach().numpy(), cmap="gray")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    ax2.imshow(y.cpu().detach().numpy(), cmap="gray")

    fig.tight_layout()
    plt.show()

def query_plots(p, x_query, y_query):
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(hspace=1 ,wspace=1)

    ax1 = fig.add_subplot(3, 5, 1)
    #ax1.title.set_text(f"Sujeto: {path_[0].split('/')[-2]}\n Slice: {slice_[0]}")
    ax1.axis("off")
    ax1.imshow(torch.where(p[0, :, :]>0.25, 1, 0).cpu().detach().numpy(), cmap="gray")

    ax12 = fig.add_subplot(3, 5, 6)
    #ax1.title.set_text(f"Sujeto: {path_[0].split('/')[-2]}\n Slice: {slice_[0]}")
    ax12.axis("off")
    ax12.imshow(x_query[0,...].squeeze().cpu().detach().numpy(), cmap="gray")

    ax13 = fig.add_subplot(3, 5, 11)
    #ax1.title.set_text(f"Sujeto: {path_[0].split('/')[-2]}\n Slice: {slice_[0]}")
    ax13.axis("off")
    ax13.imshow(y_query[0,...].squeeze().cpu().detach().numpy(), cmap="gray")

    ax2 = fig.add_subplot(3, 5, 2)
    #ax2.title.set_text('mask')
    ax2.axis("off")
    ax2.imshow(torch.where(p[1, :, :]>0.25, 1, 0).cpu().detach().numpy(), cmap="gray")

    ax22 = fig.add_subplot(3, 5, 7)
    #ax1.title.set_text(f"Sujeto: {path_[0].split('/')[-2]}\n Slice: {slice_[0]}")
    ax22.axis("off")
    ax22.imshow(x_query[1,...].squeeze().cpu().detach().numpy(), cmap="gray")

    ax23 = fig.add_subplot(3, 5, 12)
    #ax1.title.set_text(f"Sujeto: {path_[0].split('/')[-2]}\n Slice: {slice_[0]}")
    ax23.axis("off")
    ax23.imshow(y_query[1,...].squeeze().cpu().detach().numpy(), cmap="gray")

    ax3 = fig.add_subplot(3, 5, 3)
    #ax3.title.set_text('Unlabeled image')
    ax3.axis("off")
    ax3.imshow(torch.where(p[2, :, :]>0.25, 1, 0).cpu().detach().numpy(), cmap="gray") #

    ax32 = fig.add_subplot(3, 5, 8)
    #ax1.title.set_text(f"Sujeto: {path_[0].split('/')[-2]}\n Slice: {slice_[0]}")
    ax32.axis("off")
    ax32.imshow(x_query[2,...].squeeze().cpu().detach().numpy(), cmap="gray")

    ax33 = fig.add_subplot(3, 5, 13)
    #ax1.title.set_text(f"Sujeto: {path_[0].split('/')[-2]}\n Slice: {slice_[0]}")
    ax33.axis("off")
    ax33.imshow(y_query[2,...].squeeze().cpu().detach().numpy(), cmap="gray")

    ax4 = fig.add_subplot(3, 5, 4)
    #ax3.title.set_text('Unlabeled image')
    ax4.axis("off")
    ax4.imshow(torch.where(p[3, :, :]>0.25, 1, 0).cpu().detach().numpy(), cmap="gray") #

    ax42 = fig.add_subplot(3, 5, 9)
    #ax1.title.set_text(f"Sujeto: {path_[0].split('/')[-2]}\n Slice: {slice_[0]}")
    ax42.axis("off")
    ax42.imshow(x_query[3,...].squeeze().cpu().detach().numpy(), cmap="gray")

    ax43 = fig.add_subplot(3, 5, 14)
    #ax1.title.set_text(f"Sujeto: {path_[0].split('/')[-2]}\n Slice: {slice_[0]}")
    ax43.axis("off")
    ax43.imshow(y_query[3,...].squeeze().cpu().detach().numpy(), cmap="gray")

    ax5 = fig.add_subplot(3, 5, 5)
    #ax3.title.set_text('Unlabeled image')
    ax5.axis("off")
    ax5.imshow(torch.where(p[4, :, :]>0.25, 1, 0).cpu().detach().numpy(), cmap="gray") #

    ax52 = fig.add_subplot(3, 5, 10)
    #ax1.title.set_text(f"Sujeto: {path_[0].split('/')[-2]}\n Slice: {slice_[0]}")
    ax52.axis("off")
    ax52.imshow(x_query[4,...].squeeze().cpu().detach().numpy(), cmap="gray")

    ax53 = fig.add_subplot(3, 5, 15)
    #ax1.title.set_text(f"Sujeto: {path_[0].split('/')[-2]}\n Slice: {slice_[0]}")
    ax53.axis("off")
    ax53.imshow(y_query[4,...].squeeze().cpu().detach().numpy(), cmap="gray")

    fig.tight_layout()
    plt.show()

