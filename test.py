import matplotlib.pyplot as plt
import DataLoader
import os

cwd = os.getcwd()

loader = DataLoader.DataLoader(cwd + "/data/train-labels.idx1-ubyte",
                               cwd + "/data/train-images.idx3-ubyte",
                               cwd + "/data/t10k-labels.idx1-ubyte",
                               cwd + "/data/t10k-images.idx3-ubyte")

train_images, train_labels = loader.load_training()
def show_image(image, label):
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

# Display first 5 images
for i in range(5):
    show_image(train_images[i], train_labels[i])
