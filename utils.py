import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image,ImageFilter
import scipy.misc
import cv2
import numpy as np
from descreen import *

imsize = 288

loader = transforms.Compose([
             transforms.CenterCrop(imsize),
             transforms.ToTensor()
         ])

loader2 = transforms.Compose([
             transforms.Resize(size=72),               
             transforms.ToTensor()
         ])

unloader = transforms.ToPILImage()


def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    image = image.unsqueeze(0)
    return image
  
def save_images(input, paths):
    N = input.size()[0]
    # print("help me")
    images = input.data.clone().cpu()
    for n in range(N):
        image = images[n]
        image = image.view( 3,imsize, imsize)
        image = unloader(image)
        scipy.misc.imsave(paths[n], image)

def image_cutter(input):
    image=input.data.clone().cpu()
    image = image.view( 3,imsize, imsize)
    image = unloader(image)
    image=image.filter(ImageFilter.GaussianBlur(radius=1))
    image = Variable(loader2(image))
    image = image.unsqueeze(0)
    return image


def save_image(input, paths):
    image = input.data.clone().cpu()
    # image = images
    image = image.view( 3,imsize, imsize)
    image = unloader(image)
    image2=np.array(image)
    image2 = image2[:, :, ::-1].copy() 
    screen(image2,paths)
    # scipy.misc.imsave(paths, image)

def save_imager(input, paths):
    image = input.data.clone().cpu()
    # image = images
    image = image.view( 3,72, 72)
    image = unloader(image)
    image2=np.array(image)
    image2 = image2[:, :, ::-1].copy() 
    screen(image2,paths)
    # scipy.misc.imsave(paths, image)    