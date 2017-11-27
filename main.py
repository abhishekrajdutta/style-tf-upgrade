import torch.utils.data
import torchvision.datasets as datasets

from StyleCNN import *
from utils import *


# CUDA Configurations
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# Content and style
style = image_loader("testImages/style.jpg").type(dtype)
content = image_loader("testImages/content10.jpg").type(dtype)
pastiche = image_loader("testImages/content10.jpg").type(dtype)
pastiche.data = torch.randn(pastiche.data.size()).type(dtype)

num_epochs = 3
N = 1
# print("LETSGO")
kwargs = {'num_workers': 1} if torch.cuda.is_available() else {}

if args.mode=="train":
	train=1
	test=0
elif args.mode=="test":
	train=0
	test=1

# train=0
# test=1	

def main(style,kwargs):
	
	if train==1:
		style_cnn = StyleCNN(style)

		

		# Contents
		# coco = datasets.ImageFolder(root='images/coco/', transform=loader)
		faces = datasets.ImageFolder(root='images/faces/', transform=loader)
		content_loader = torch.utils.data.DataLoader(faces, batch_size=N, shuffle=True,drop_last=True,pin_memory=True, **kwargs)

		for epoch in range(num_epochs):
			for i, content_batch in enumerate(content_loader):
				# print(content_batch)				
				iteration = epoch * i + i
				content_batch[0]=content_batch[0].type(dtype)
				
				content_loss, style_loss, pastiches = style_cnn.train(content_batch[0])

				if i % 10 == 0:
				  print("Iteration: %d" % (iteration))
				  print("Content loss: %f" % (content_loss.data[0]))
				  print("Style loss: %f" % (style_loss.data[0]))

				if i % 500 == 0:
				  path = "outputs/%d_" % (iteration)
				  paths = [path + str(n) + ".png" for n in range(N)]
				  # print(pastiches.size())				
				  save_images(pastiches, paths)

				  path = "outputs/content_%d_" % (iteration)
				  paths = [path + str(n) + ".png" for n in range(N)]
				  # save_images(content_batch[0], paths) ##TODO save content images
				  # style_cnn.save() ##save models

				  modelname="models/it%d.pt" % (iteration)
				  torch.save(style_cnn, modelname)

	if test==1:
		# style_cnn = StyleCNN(style)
		style_cnn = torch.load(args.model)
		# style_cnn = torch.load("models/it3500.pt")
		pastiche=style_cnn.test(content)
		path = "outputs/trained.png"
		save_image(pastiche, path)






main(style,kwargs)