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
from gan_params import input_size_generator
from gan_params import input_size_discriminator
from gan_params import learning_rate
from gan_params import batch_size
from gan_params import seq_length
from gan_params import latent_dim
from gan_params import num_generated_features
from gan_params import num_layers


torch.manual_seed(1)

'''
The GAN genrator class with all the parameters defining the generator lstm
'''

class GANGenerator(nn.Module):
	def __init__(self,hidden_units):	

		super(GANGenerator, self).__init__()
		self.hidden_units = hidden_units
		self.input_size = input_size_generator
		self.batch_size = batch_size
		self.latent_dim = latent_dim
		self.seq_length = seq_length
		self.num_generated_features = num_generated_features
		self.num_layers = num_layers

		self.W_out_g = self.init_GeneratorWeights()
		self.b_out_g = self.init_GeneratorBiase()


		self.generator = nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_units,num_layers= self.num_layers).to(device)


	def init_GeneratorWeights(self):
		temp = torch.distributions.Normal(0,1).sample((self.hidden_units,self.num_generated_features)).to(device)
		temp.requires_grad = True
		# print temp.requires_grad
		return temp

	def init_GeneratorBiase(self):
		temp = torch.distributions.Normal(0,1).sample([1]).to(device)
		temp.requires_grad = True
		return temp


	'''
	Forward pass of the generator with one simple layer added
	The return of the function is a tensor of batchsize, time series length and generated features
	'''

	def forward(self,Z):
		lstm_out,_ = self.generator(Z.view(self.seq_length,-1,self.input_size))
		logits_2d = torch.matmul(lstm_out,self.W_out_g) + self.b_out_g
		output_2d = torch.tanh(logits_2d)
		output_3d = output_2d.view(-1,self.seq_length,self.num_generated_features)
		return output_3d


'''
The gan discriminator model with all the parameters defining the lstm
'''
class GANDiscriminator(nn.Module):
	def __init__(self,hidden_units):
		
		super(GANDiscriminator,self).__init__()
		self.hidden_units = hidden_units
		self.input_size_discriminator = input_size_discriminator
		self.seq_length = seq_length
		self.num_generated_features = num_generated_features
		self.num_layers = num_layers

		self.W_out_d = self.init_DiscriminatorWeights()
		self.b_out_d = self.init_DiscriminatorBiase()
		self.discriminator = nn.LSTM(input_size=self.num_generated_features,hidden_size=self.hidden_units,num_layers=self.num_layers).to(device)


	def init_DiscriminatorWeights(self):
		temp = torch.distributions.Normal(0,1).sample([self.hidden_units,1]).to(device)
		temp.requires_grad = True
		return temp

	def init_DiscriminatorBiase(self):
		temp = torch.distributions.Normal(0,1).sample([1]).to(device)
		temp.requires_grad = True
		return temp


	def forward(self,data):
		lstm_out,_ = self.discriminator(data.view(self.seq_length,-1,self.num_generated_features))
		lstm_out_flat = lstm_out.view(-1,self.hidden_units)
		logits = torch.matmul(lstm_out_flat, self.W_out_d) + self.b_out_d
		output = torch.sigmoid(logits)

		return output,logits



'''
Testing the implementation of the generator and discriminator
'''

if __name__ == '__main__':
	x = torch.randn((10,10,34))
	generator = GANGenerator(hidden_units)
	output = generator.forward(x)	