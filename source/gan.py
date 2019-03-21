import numpy as np
import pandas as pd 
import math
import torch
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from gan_params.py import hidden_units
from gan_params.py import state_is_tuple
from gan_params.py import input_size

torch.manual_seed(1)


class GANGenerator(nn.Module):
	def __init__(self,hidden_units,state_is_tuple):	

		super(GANGenerator, self).__init__()
		self.hidden_units = hidden_units
		self.input_size = input_size

		self.generator = nn.LSTM(input_size=input_size,hidden_size=self.hidden_units,num_layers=1).to(device)