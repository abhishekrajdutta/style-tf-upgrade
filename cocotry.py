from utils import *
import torch

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

content = image_loader("testImages/style.jpg").type(dtype)

save_image(content, 'test.png')