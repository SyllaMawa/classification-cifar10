import matplotlib.pyplot as plt
import numpy as np
from dataset import *

def imshow(img):
    img = img / 2 + 0.5
    img_np = img.numpy()
    print(f"Shape de l'image : {img_np.shape}") # C H W
    plt.imshow(np.transpose(img_np, (1, 2, 0)))
    plt.show()

#random img
data_iter = iter(trainloader)
imgs, labels = next(data_iter)

#display
imshow(torchvision.utils.make_grid(imgs))

#print label
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))