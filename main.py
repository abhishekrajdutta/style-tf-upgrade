import torch.utils.data
import torchvision.datasets as datasets

from StyleCNN import *
from utils import *

# CUDA Configurations
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# Content and style
style = image_loader("testImages/picasso.jpg").type(dtype)
content = image_loader("testImages/dancing.jpg").type(dtype)
pastiche = image_loader("testImages/dancing.jpg").type(dtype)
pastiche.data = torch.randn(pastiche.data.size()).type(dtype)

num_epochs = 3
N = 4
# print("LETSGO")
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

def main(style,kwargs):
	style_cnn = StyleCNN(style)

	

	# Contents
	coco = datasets.ImageFolder(root='images/', transform=loader)
	content_loader = torch.utils.data.DataLoader(coco, batch_size=N, shuffle=True,drop_last=True, **kwargs)

	for epoch in range(num_epochs):
		for i, content_batch in enumerate(content_loader):
			# print(content_batch)				
			iteration = epoch * i + i
			content_loss, style_loss, pastiches = style_cnn.train(content_batch[0])

			if i % 10 == 0:
			  print("Iteration: %d" % (iteration))
			  print("Content loss: %f" % (content_loss.data[0]))
			  print("Style loss: %f" % (style_loss.data[0]))

			if i % 500 == 0:
			  path = "outputs/%d_" % (iteration)
			  paths = [path + str(n) + ".png" for n in range(N)]
			  save_images(pastiches, paths)

			  path = "outputs/content_%d_" % (iteration)
			  paths = [path + str(n) + ".png" for n in range(N)]
			  save_images(content_batch, paths)
			  style_cnn.save()


main(style,kwargs)