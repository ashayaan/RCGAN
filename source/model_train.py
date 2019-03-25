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

class Network(nn.module):
	def __init__(self):
		