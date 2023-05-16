from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Subset, DataLoader, Dataset
import numpy as np
import os, shutil


import tkinter as tk
from PIL import ImageTk, Image

import augmax_modules.corruptions_IN


def show_imge(path):
    image_window = tk.Tk()
    img = ImageTk.PhotoImage(Image.open(path))
    panel = tk.Label(image_window, image=img)
    panel.pack(side="bottom", fill="both", expand="yes")
    image_window.mainloop()

#show_imge("/home/jessicalnewman2001/Documents/GitHub/AugMax/fishman.JPEG")
img = Image.open("/home/jessicalnewman2001/Documents/GitHub/AugMax/1.jpg")
imgdata = np.asarray(img)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

preprocess = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean, std)])

train_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor()]) # if transform_train else transforms.Compose(
            # [transforms.Resize(256),
            #  transforms.CenterCrop(224),
            #  preprocess])
test_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             preprocess])



transformed = train_transform(img)

#print(transformed)

convert = transforms.ToPILImage()
image1 = convert(transformed)
image1.show()

transf = augmax_modules.corruptions_IN.gaussian_blur(image1,5)
print(transf)
Image.fromarray((transf * 1).astype(np.uint8)).convert('RGB').show()

spattered = augmax_modules.corruptions_IN.spatter(image1, 5)
Image.fromarray((spattered * 1).astype(np.uint8)).convert('RGB').show()

fogged = augmax_modules.corruptions_IN.fog(image1, 4)
Image.fromarray((fogged * 1).astype(np.uint8)).convert('RGB').show()

noised = augmax_modules.corruptions_IN.gaussian_noise(image1,5)
Image.fromarray((noised * 1).astype(np.uint8)).convert('RGB').show()

motioned = augmax_modules.corruptions_IN.motion_blur(image1,5)
Image.fromarray((motioned * 1).astype(np.uint8)).convert('RGB').show()

zoomed = augmax_modules.corruptions_IN.zoom_blur(image1,5)
Image.fromarray((zoomed * 1).astype(np.uint8)).convert('RGB').show()

frosted = augmax_modules.corruptions_IN.frost(image1,4)
Image.fromarray((frosted * 1).astype(np.uint8)).convert('RGB').show()

elasticed = augmax_modules.corruptions_IN.elastic_transform(image1,5)
Image.fromarray((elasticed * 1).astype(np.uint8)).convert('RGB').show()

pixeled = augmax_modules.corruptions_IN.pixelate(image1,5)
pixeled.show()

        # test_data = datasets.ImageFolder(validation_root, transform=test_transform)
