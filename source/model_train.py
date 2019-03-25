import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import math

from gan_params import hidden_units
from gan_params import input_size_generator
from gan_params import input_size_discriminator
from gan_params import learning_rate
from gan_params import batch_size
from gan_params import seq_length
from gan_params import latent_dim
from gan_params import num_generated_features


from gan import (GANGenerator, GANDiscriminator)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
	def __init__(self,hidden_units):
		
		super(Network,self).__init__()

		self.hidden_units = hidden_units
		self.learning_rate = learning_rate
		self.generator = GANGenerator(self.hidden_units)
		self.discriminator = GANDiscriminator(self.hidden_units)

		#GAN generator parameters
		generator_model_parameters =  list(self.generator.parameters())
		generator_model_parameters.extend([self.generator.W_out_g, self.generator.b_out_g])
		
		#GAN discriminator parameters
		discriminator_model_parameters = list(self.discriminator.parameters())
		discriminator_model_parameters.extend([self.discriminator.W_out_d, self.discriminator.b_out_d])

		'''
		Defining the optimizer for the GAN
		The generator will use a Adam optimizer
		The discriminator will use a SGD optimizer
		'''
		self.generatorOptimizer = optim.Adam(generator_model_parameters)
		self.discriminatorOPtimizer = optim.SGD(discriminator_model_parameters, lr = self.learning_rate)



if __name__ == '__main__':
	test = Network(hidden_units)

