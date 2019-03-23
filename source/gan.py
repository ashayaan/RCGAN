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


from gan_params import hidden_units
from gan_params import input_size
from gan_params import learning_rate
from gan_params import batch_size
from gan_params import seq_length
from gan_params import latent_dim
from gan_params import num_generated_features


torch.manual_seed(1)

'''
The GAN genrator class with all the parameters defining the generator lstm
'''

class GANGenerator(nn.Module):
	def __init__(self,hidden_units):	

		super(GANGenerator, self).__init__()
		self.hidden_units = hidden_units
		self.input_size_generator = input_size
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.latent_dim = latent_dim
		self.seq_length = seq_length
		self.num_generated_features = num_generated_features

		self.W_out_g = self.init_weights()
		self.b_out_g = self.init_biase()

		self.generator = nn.LSTM(input_size=input_size,hidden_size=self.hidden_units,num_layers=1).to(device)


	# def sample_Z(self):
	# 	sample = np.float32(np.random.normal(size=[self.batch_size, self.seq_length, self.latent_dim]))
	# 	return sample

	def init_weights(self):
		return torch.distributions.Normal(0,1).sample((self.hidden_units,self.num_generated_features)).to(device)

	def init_biase(self):
		return torch.distributions.Normal(0,1).sample([1])


	'''
	Forward pass of the generator with one simple layer added
	The return of the function is a tensor of batchsize, time series length and generated features
	'''

	def forward(self,Z):
		lstm_out,_ = self.generator(Z.view(self.seq_length,self.batch_size,-1))
		logits_2d = torch.matmul(lstm_out,self.W_out_g) + self.b_out_g
		output_2d = nn.Tanh(logits_2d)
		output_3d = output_2d.view(-1,self.seq_length,self.num_generated_features)
		return output_3d


'''
The gan discriminator model with all the parameters defining the lstm
'''
class GANdiscriminator(nn.Module):
	def __init__(self,hidden_units):
		
		super(GANdiscriminator,self).__init__()
		self.hidden_units = hidden_units
