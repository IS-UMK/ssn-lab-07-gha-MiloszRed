import matplotlib.pyplot as plt

def plot_gallery(images, h, w, titles=None ,n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(min(n_row * n_col, len(images))):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        if titles is not None: 
            plt.title(titles[i], size=12)
        else:
            plt.title(str(i+1), size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()