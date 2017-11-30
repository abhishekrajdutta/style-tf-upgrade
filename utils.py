import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
from PIL import Image,ImageFilter
import scipy.misc
import cv2
import numpy as np
from descreen import *


imsize = 288
f=4

loader = transforms.Compose([
             transforms.CenterCrop(imsize),
             transforms.ToTensor()
         ])

loader2 = transforms.Compose([
             transforms.Resize(size=imsize/f),               
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
    N = input.size()[0]
    images = input.data.clone().cpu()
    imagetray = torch.FloatTensor(N,3,(imsize/f),(imsize/f))
    for n in range(N):
        image = images[n]
        image = image.view( 3,imsize, imsize)
        image = unloader(image)
        image=image.filter(ImageFilter.GaussianBlur(radius=1))
        image = loader2(image)
        imagetray[n] = image
    return imagetray


def save_image(input, paths):
    image = input.data.clone().cpu()
    # image = images
    image = image.view( 3,imsize, imsize)
    image = unloader(image)
    scipy.misc.imsave(paths, image)
    # image2=np.array(image)
    # image2 = image2[:, :, ::-1].copy() 
    # screen(image2,paths)
    # scipy.misc.imsave(paths, image)

def save_imager(input, paths):
    image = input.data.clone().cpu()
    # image = images
    image = image.view( 3,imsize/f, imsize/f)
    image = unloader(image)
    scipy.misc.imsave(paths, image)
    # image2=np.array(image)
    # image2 = image2[:, :, ::-1].copy() 
    # screen(image2,paths)
    # scipy.misc.imsave(paths, image)    

def save_image_out(input, paths):
    image = input.data.clone().cpu()
    # image = images
    image = image.view( 3,imsize*f, imsize*f)
    image = unloader(image)
    scipy.misc.imsave(paths, image)
    # image2=np.array(image)
    # image2 = image2[:, :, ::-1].copy() 
    # screen(image2,paths)
    # scipy.misc.imsave(paths, image)  