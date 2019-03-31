import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import math
import argparse

from gan_params import hidden_units
from gan_params import input_size_generator
from gan_params import input_size_discriminator
from gan_params import learning_rate
from gan_params import batch_size
from gan_params import seq_length
from gan_params import latent_dim
from gan_params import num_generated_features
from gan_params import num_epochs
from gan_params import G_rounds
from gan_params import D_rounds
from torch.distributions import normal



from gan import (GANGenerator, GANDiscriminator)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
	def __init__(self,hidden_units):
		
		super(Network,self).__init__()

		self.hidden_units = hidden_units
		self.learning_rate = learning_rate
		self.seq_length = seq_length
		self.batch_size = batch_size
		self.latent_dim = latent_dim
		self.G_rounds = G_rounds
		self.D_rounds = D_rounds

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
		self.discriminatorOptimizer = optim.SGD(discriminator_model_parameters, lr = self.learning_rate, momentum=0.4)


	def smaple_Z(self):
		m = normal.Normal(0,1)
		return m.sample((self.batch_size,self.seq_length,self.latent_dim))

	#Calculate the generator loss
	def generatorLoss(self,logits,target):
		loss = nn.BCELoss(reduction='mean')
		return loss(logits,target)
	

	#Calculate the discriminator loss
	def discriminatorLoss(self,logits,target):
		loss = nn.BCELoss(reduction='mean')
		return loss(logits,target)
		

''' 
Reading the real stock data
'''
def readData(file):
	df = pd.read_csv(file)
	df = df.drop(columns = ['Date', 'Vol.','Change %']).values.astype(dtype=np.float32())
	df = df.reshape((-1,seq_length,num_generated_features))
	return df

'''
Getting the mini-batches to train the discriminator and returns a tensor
'''
def get_batch(data,batch_size, batch_index):
	start_pos = batch_size * batch_index
	end_pos = start_pos + batch_size
	return torch.from_numpy(data[start_pos:end_pos])


'''
Function to the GAN
'''

def train_network(network,data,batch_size):
	for batch_index in range(0, int(len(data)/batch_size) - (network.D_rounds + network.G_rounds), network.D_rounds + network.G_rounds): 
		X = get_batch(data, batch_size, batch_index)
		Z = network.smaple_Z()

		#Froward Pass of the GAN
		G_sample = network.generator.forward(Z)
		D_real, D_logits_real = network.discriminator.forward(X)
		D_fake, D_logits_fake = network.discriminator.forward(G_sample)

		# print G_sample

		#loss calculation
		G_loss = network.generatorLoss(D_fake, torch.ones_like(D_fake))
		D_loss_fake = network.discriminatorLoss(D_fake, torch.zeros_like(D_fake))
		D_loss_real = network.discriminatorLoss(D_real, torch.ones_like(D_real))
		D_loss = D_loss_real + D_loss_fake

		print("GENERATOR LOSS: {}".format(G_loss.item()))
		print("DISCRIMINATOR LOSS: {}".format(D_loss.item())) 

		#Back propagation
		network.generatorOptimizer.zero_grad()
		network.discriminatorOptimizer.zero_grad()

		G_loss.backward(retain_graph=True)
		network.generatorOptimizer.step()

		D_loss.backward(retain_graph=True)
		network.discriminatorOptimizer.step()

	return network


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--datapath", type=str, default="../data/daily_data", help="path to the dataset")
	parser.add_argument("--file", type=str, default="BHEL_Historical_Data.csv")
	args = parser.parse_args()

	file = args.datapath + '/' + args.file
	data = readData(file)



	network = Network(hidden_units)

	for epoch in range(num_epochs):
		print ('EPOCH : {}'.format(epoch + 1))
		network = train_network(network, data, batch_size)

	print network.generator.forward(network.smaple_Z())