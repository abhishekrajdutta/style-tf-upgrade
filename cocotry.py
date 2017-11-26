from utils import *
import torch
import torchvision.datasets as datasets
import torch.utils.data
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
N=4
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
coco = datasets.ImageFolder(root='images/', transform=loader)
content_loader = torch.utils.data.DataLoader(coco, batch_size=N, shuffle=True,drop_last=True, **kwargs)


class GramMatrix(nn.Module):
    
    def forward(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        
        return G.div(a * b * c * d)



print(content_loader)
# print(enumerate(content_loader))
for i, content_batch in enumerate(content_loader):
	# content_loss, style_loss, pastiches = train(content_batch[0])
   
	content_layers = ['conv_4']
	style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
	content_weight = 1
	style_weight = 1000
	loss_network = models.vgg19(pretrained=True)
	transform_network = nn.Sequential(nn.ReflectionPad2d(40),
										   nn.Conv2d(3, 32, 9, stride=1, padding=4),
										   nn.Conv2d(32, 64, 3, stride=2, padding=1),
										   nn.Conv2d(64, 128, 3, stride=2, padding=1),
										   nn.Conv2d(128, 128, 3, stride=1, padding=0),
										   nn.Conv2d(128, 128, 3, stride=1, padding=0),
										   nn.Conv2d(128, 128, 3, stride=1, padding=0),
										   nn.Conv2d(128, 128, 3, stride=1, padding=0),
										   nn.Conv2d(128, 128, 3, stride=1, padding=0),
										   nn.Conv2d(128, 128, 3, stride=1, padding=0),
										   nn.Conv2d(128, 128, 3, stride=1, padding=0),
										   nn.Conv2d(128, 128, 3, stride=1, padding=0),
										   nn.Conv2d(128, 128, 3, stride=1, padding=0),
										   nn.Conv2d(128, 128, 3, stride=1, padding=0),
										   nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
										   nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
										   nn.Conv2d(32, 3, 9, stride=1, padding=4),
										   )
	gram = GramMatrix()
	loss = nn.MSELoss()
	optimizer = optim.Adam(transform_network.parameters(), lr=1e-3)
	
	optimizer.zero_grad()