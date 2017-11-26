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

num_epochs = 31

def main(style,content,pastiche):

	style_cnn = StyleCNN(style, content, pastiche)
	
	for i in range(num_epochs):
		pastiche = style_cnn.train()
	
		if i % 10 == 0:
			print("Iteration: %d" % (i))
			
			path = "outputs/%d.png" % (i)
			pastiche.data.clamp_(0, 1)
			save_image(pastiche, path)

main(style,content,pastiche)