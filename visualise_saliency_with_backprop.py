# Uncovering what neural nets “see” with FlashTorch
# Misa Ogura

# pip install flashtorch

import matplotlib.pyplot as plt

import torch
import torchvision.models as models

from flashtorch.utils import (apply_transforms,
                              denormalize,
                              format_for_plotting,
                              load_image,
                              visualize)

from flashtorch.utils import ImageNetIndex

from flashtorch.saliency import Backprop

image = load_image("D:/Projects Machine Learning/Data/Some from ImageNet/tabby cat.jpg")

plt.imshow(image)
plt.title('Original image')
plt.axis('off');

# next is needed, otherwise no image pops up
plt.waitforbuttonpress()

model = models.alexnet(pretrained=True)
backprop = Backprop(model)

imagenet = ImageNetIndex()
target_class = imagenet['tabby cat']

input_ = apply_transforms(image)

# Calculate the gradients of each pixel w.r.t. the input image

gradients = backprop.calculate_gradients(input_, target_class)

# Or, take the maximum of the gradients for each pixel across colour channels.

max_gradients = backprop.calculate_gradients(input_, target_class, take_max=True)

print('Shape of the gradients:', gradients.shape)
print('Shape of the max gradients:', max_gradients.shape)

visualize(input_, gradients, max_gradients)
plt.waitforbuttonpress()

backprop = Backprop(model)

guided_gradients = backprop.calculate_gradients(input_, target_class, guided=True)
max_guided_gradients = backprop.calculate_gradients(input_, target_class, take_max=True, guided=True)

visualize(input_, guided_gradients, max_guided_gradients, alpha=0.7)
plt.waitforbuttonpress()
