import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
from GramMatrix import *
from UpsampleConvLayer import *
from histmatch import *
from chainer import cuda, optimizers, serializers
from chainer import Variable as vb
import chainer.functions as F

import numpy as np

from utils import *


dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class ResCNN(object):
    def __init__(self):
        super(ResCNN, self).__init__()
        
        self.content_layers = ['conv_4']
        self.content_weight = 1
        self.style_weight = 1000
        self.loss_network = models.vgg16(pretrained=True)

        
        
        self.transform_network = nn.Sequential(nn.ReflectionPad2d(8),
                                               nn.Conv2d(3, 64, 9, stride=1, padding=4),
                                               nn.InstanceNorm2d(64, affine=True),
                                               nn.ReLU(),
                                               
                                               nn.Conv2d(64, 64, 3, stride=1, padding=0),
                                               nn.InstanceNorm2d(64, affine=True),
                                               nn.ReLU(),
                                               nn.Conv2d(64, 64, 3, stride=1, padding=0),
                                               nn.InstanceNorm2d(64, affine=True),

                                               nn.Conv2d(64, 64, 3, stride=1, padding=0),
                                               nn.InstanceNorm2d(64, affine=True),
                                               nn.ReLU(),
                                               nn.Conv2d(64, 64, 3, stride=1, padding=0),
                                               nn.InstanceNorm2d(64, affine=True),

                                               nn.Conv2d(64, 64, 3, stride=1, padding=0),
                                               nn.InstanceNorm2d(64, affine=True),
                                               nn.ReLU(),
                                               nn.Conv2d(64, 64, 3, stride=1, padding=0),
                                               nn.InstanceNorm2d(64, affine=True),

                                               nn.Conv2d(64, 64, 3, stride=1, padding=0),
                                               nn.InstanceNorm2d(64, affine=True),
                                               nn.ReLU(),
                                               nn.Conv2d(64, 64, 3, stride=1, padding=0),
                                               nn.InstanceNorm2d(64, affine=True),

                                               
                                              
                                               UpsampleConvLayer(64, 64, kernel_size=3, stride=1, upsample=2),
                                               nn.InstanceNorm2d(64, affine=True),
                                               nn.ReLU(),

                                               UpsampleConvLayer(64, 64, kernel_size=3, stride=1, upsample=2),
                                               nn.InstanceNorm2d(64, affine=True),
                                               nn.ReLU(),
                                               
                                               nn.Conv2d(64, 3, 9, stride=1, padding=4),
                                               nn.ReLU()
                                               )
        self.gram = GramMatrix()
        self.loss = nn.MSELoss()
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.loss_network=self.loss_network.cuda()
            self.loss=self.loss.cuda()
            self.gram=self.gram.cuda()
            self.transform_network=self.transform_network.cuda()
        self.optimizer = optim.Adadelta(self.transform_network.parameters(), lr=1e-3)

    
        

    def total_variation(self,x):
      xp = cuda.get_array_module(x)
      b, ch, h, w = x.shape
      wh = vb(xp.asarray([[[[1], [-1]], [[0], [0]], [[0], [0]]], [[[0], [0]], [[1], [-1]], [[0], [0]]], [[[0], [0]], [[0], [0]], [[1], [-1]]]], dtype=np.float32))
      ww = vb(xp.asarray([[[[1, -1]], [[0, 0]], [[0, 0]]], [[[0, 0]], [[1, -1]], [[0, 0]]], [[[0, 0]], [[0, 0]], [[1, -1]]]], dtype=np.float32))
      return F.sum(F.convolution_2d(x, W=wh) ** 2).data + F.sum(F.convolution_2d(x, W=ww) ** 2).data
      
        
    def train(self, content):
        self.optimizer.zero_grad()        
        content = Variable(content.clone().type(dtype))
        # contect.data.clamp_(0, 1)
        # save_image(content,"content.png")
        contentlow=image_cutter(content).type(dtype)
        # save_imager(contentlow,"contentlow.png")
        pastiche = self.transform_network.forward(contentlow)
        # save_image(pastiche,"past.png")
        pastiche1=pastiche.clone()
        content_loss = 0
        variation_loss=0

        i = 1
        not_inplace = lambda layer: nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
        for layer in list(self.loss_network.features):
            layer = not_inplace(layer)
            if self.use_cuda:
                layer=layer.cuda()

            pastiche, content = layer.forward(pastiche), layer.forward(content)
           


            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)

                if name in self.content_layers:
                    content_loss += self.loss(pastiche * self.content_weight, content.detach() * self.content_weight)
                


            if isinstance(layer, nn.ReLU):
                i += 1


        pastiche2=pastiche1.data.cpu().numpy()
        total_loss = content_loss #+ (0.5e-5)*self.total_variation(pastiche2)
        total_loss.backward()
        self.optimizer.step()

        return content_loss,pastiche1


    def test(self, content):
    	pastiche = self.transform_network.forward(content)
    	return pastiche

    