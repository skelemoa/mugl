import torch 
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import sys


class BasicBlock(nn.Module):
	"""
	Basic block is composed of 2 CNN layers with residual connection.
	Each CNN layer is followed by batchnorm layer and swish activation 
	function. 
	Args:
		in_channel: number of input channels
		out_channel: number of output channels
		k: (default = 1) kernel size
	"""
	def __init__(self, in_channel, out_channel, k=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(
			in_channel,
			out_channel,
			kernel_size=k,
			padding=(0, 0),
			stride=(1, 1))
		self.bn1 = nn.BatchNorm2d(out_channel)

		self.conv2 = nn.Conv2d(
			out_channel,
			out_channel,
			kernel_size=1,
			padding=(0, 0),
			stride=(1, 1))
		self.bn2 = nn.BatchNorm2d(out_channel)

		self.shortcut = nn.Sequential()
		# if in_channel != out_channel:
		self.shortcut.add_module(
			'conv',
			nn.Conv2d(
				in_channel,
				out_channel,
				kernel_size=k,
				padding=(0,0),
				stride=(1,1)))
		self.shortcut.add_module('bn', nn.BatchNorm2d(out_channel))

	def swish(self,x):
		"""
		We use swish in spatio-temporal encoding/decoding. We tried with 
		other activation functions such as ReLU and LeakyReLU. But we 
		achieved the best performance with swish activation function.
		Args:
			X: tensor: (batch_size, ...)
		Return:
			_: tensor: (batch, ...): applies swish 
			activation to input tensor and returns  
		"""
		return x*torch.sigmoid(x)

	def forward(self, x):
		y = self.swish(self.bn1(self.conv1(x)))
		y = self.swish(self.bn2(self.conv2(y)))
		y = y + self.shortcut(x)
		y = self.swish(y)
		return y



class Model(nn.Module):
	"""
	Model for our proposed GMVAE model for human motion generation. It is composed
	of 3 encoder modules and 3 decoder modules.
	Args:
		num_class: number of action categories
		latent_dim: bottleneck dimension of the autoencoder
		components: number of Gaussian components used for sampling. We keep the
		number of components same as the number of classes
	"""
	def __init__(self, num_class, latent_dim, components=120):
		super(Model, self).__init__()
		self.latent_dim = latent_dim

		# encoder
		self.encoder1 = BasicBlock(1,1,k=5)
		self.encode_t = BasicBlock(64, 32)
		self.encode_t1 = BasicBlock(32, 8)
		self.encode_t2 = BasicBlock(8, 4)

		# decoder		
		self.conv1 = BasicBlock(1,1)
		self.decode_t = BasicBlock(4,8)
		self.decode_t1 = BasicBlock(8,32)
		self.decode_t2 = BasicBlock(32,64)
		self.decoder = nn.Linear(88, 144*2)

		# Reparameterization
		self.mean = nn.Linear(352, self.latent_dim)
		self.logvar = nn.Linear(352, self.latent_dim)
		self.hidden2latent = nn.Linear(352+20+num_class+6+4, 352*2)
		self.latent2hidden = nn.Linear(self.latent_dim+num_class+6, 352)

		# root trajectory
		self.root1 = nn.Conv1d(4,8,5)
		self.root2 = nn.Conv1d(8,16,5)
		self.root3 = nn.Conv1d(16,32,5)
		self.root4 = nn.Conv1d(32,64,5)
		self.root5 = nn.Linear(72,3*2)



		# root trajectory encoder
		self.r_encoder0 = nn.Conv1d(64, 32, 1)
		self.r_encoder1 = nn.Conv1d(32, 16, 1)
		self.r_encoder2 = nn.Conv1d(16, 8, 1)
		self.r_encoder3 = nn.Conv1d(8, 4, 1)
		self.r_encoder4 = nn.Linear(24, 20)
		# self.r_encoder4 = nn.Linear(32, 16, 5)


		# sequence length encoder
		self.seq_encoder0 = nn.Conv1d(64,32,1)
		self.seq_encoder1 = nn.Conv1d(32,16,1)
		self.seq_encoder2 = nn.Conv1d(16,8,1)
		self.seq_encoder3 = nn.Conv1d(8,4,1)
		self.seq_encoder4 = nn.Linear(4, 4)

		# sequence length decoder
		# decoder
		self.seq_decoder1 = nn.Linear(latent_dim, 4)
		self.seq_decoder2 = nn.Conv1d(4,8,1)
		self.seq_decoder3 = nn.Conv1d(8,16,1)
		self.seq_decoder4 = nn.Conv1d(16,32,1)
		self.seq_decoder5 = nn.Conv1d(32,64,1)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()


		# Gausiam mixture parameters
		self.components = components

		# mixture of Gaussian parameters
		self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.components, self.latent_dim)
                                        / np.sqrt(self.components * self.latent_dim))
		
		# Uniform weighting
		self.pi = torch.nn.Parameter(torch.ones(components) / components, requires_grad=False)


	def swish(self,x):
		"""
		We use swish in spatio-temporal encoding/decoding. We tried with 
		other activation functions such as ReLU and LeakyReLU. But we 
		achieved the best performance with swish activation function.
		Args:
			X: tensor: (batch_size, ...)
		Return:
			_: tensor: (batch, ...): applies swish 
			activation to input tensor and returns  
		"""
		return x*torch.sigmoid(x)


	def encoder_net(self, X):
		"""
		Encoder first downsamples the input motion in the spatial dimension
		and then downsamples in the temporal dimension and returns spatio-
		temporal feature.
		Args:
			X: tensor: (batch_size, 64, 48, 6): input motion of 2 persons. 24
			joints for each persons so total 48 joints.
		Return:
			x: tensor: (batch_size, 4, ...): spatio-temporal feature 
		"""
		N,T,J = X.shape
		# pose encoding
		x = X.reshape((N*T,1,48,6))
		x = self.encoder1(x)

		# temporal encoding
		x = x.reshape((N,T,44,2))
		x = self.encode_t(x)
		x = self.encode_t1(x)
		x = self.encode_t2(x)
		x = x.reshape((N,4,-1))

		return x


	def decoder_net(self, X):
		"""
		The deocder is opposit of the encoder. It takes the vector sampled
		from a mixture of gaussian parameter conditioned by class label on-
		hot vector and viewpoint vector, upsamples it in the temporal dimension 
		first and then upsamples it in the spatial dimension.
		Args:
			X: tensor: (batch_size, 4, ...): sampled vector conditionied on class 
			label and viewpoint
		Return:
			x: tensor: (batch_size, 64, 48, 6): generated human motion
		"""
		N,T,J = X.shape
		# temporal decoding
		x = X.reshape((N,T,44,2))
		x = self.decode_t(x)
		x = self.decode_t1(x)
		x = self.decode_t2(x)

		# pose decoding
		x = x.reshape((N*64,1,44,2))
		x = self.conv1(x)
		x = x.reshape((N,64, -1))
		x = self.decoder(x)

		return x


	def root_traj(self, z):
		"""
		This function calculate the root trajectory for 2-person inteaction
		classes. generates the displacement of the second person's root from the
		first person's root.
		Args:
			z: tensor: (batch_size, 4, ...): sampled vector conditionied on class 
			label and viewpoint
		Return:
			z: tensor: (batch_size, 64, 3): displacement
		"""
		z = self.swish(self.root1(z))
		z = self.swish(self.root2(z))
		z = self.swish(self.root3(z))
		z = self.swish(self.root4(z))
		z = self.root5(z)
		
		return z



	def root_traj_encoder(self, root):
		root = root.float()
		z = self.swish(self.r_encoder0(root))
		z = self.swish(self.r_encoder1(z))
		z = self.swish(self.r_encoder2(z))
		z = self.swish(self.r_encoder3(z))
		z = z.reshape((z.shape[0], -1))
		z = self.r_encoder4(z)
		
		return z


	def seq_encoder(self, x):
		N,T,_ = x.shape
		z = self.relu(self.seq_encoder0(x))
		z = self.relu(self.seq_encoder1(z))
		z = self.relu(self.seq_encoder2(z))
		z = self.relu(self.seq_encoder3(z))
		z = z.reshape((N, -1))
		z = self.relu(self.seq_encoder4(z))
		return z

	def seq_decoder(self, z):
		N,_ = z.shape
		z = self.relu(self.seq_decoder1(z))
		z = z.unsqueeze(-1)
		z = self.relu(self.seq_decoder2(z))
		z = self.relu(self.seq_decoder3(z))
		z = self.relu(self.seq_decoder4(z))
		z = self.seq_decoder5(z)
		z = self.sigmoid(torch.cumsum(z.squeeze(2), dim=1).unsqueeze(2)) # monotonically increasing sequence

		return z



	def reparameterization(self,mean,logvar):
		"""
		this function sample from a unimodal Gaussian distribution in case of 
		vanilla VAE.
		Args: 
			mean: tensor: (batch_size, ...): mean
			logvar: tensor: (batch_size, ...): var
		Return:
			_: tensor: (batch_size, ...): sampled latent vector
		"""
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std).to(logvar.device)
		return eps*std + mean


	def forward(self, x, y, rot, root, seq):
		x = self.encoder_net(x.float())
		seq = self.seq_encoder(seq)
		root_encoding = self.root_traj_encoder(root)
		# reparameterization
		z = x.reshape((x.shape[0], -1))
		z = torch.cat((z,root_encoding, seq, y.float(), rot.float()), dim=1)
		z = self.hidden2latent(z)
		# mean = self.mean(z)
		# logvar = self.logvar(z)
		mean, var = self.gaussian_parameters(z, dim=1)
		# print(mean.shape, var.shape)

		# Gaussian mixture

		prior = self.gaussian_parameters(self.z_pre, dim=1)

		z = self.sample_gaussian(mean, var)

		# terms for KL divergence
		log_q_phi = self.log_normal(z, mean, var)
		log_p_theta = self.log_normal_mixture(z, prior[0], prior[1])
		kld = torch.mean(log_q_phi - log_p_theta)

		# z = self.reparameterization(mean, logvar)
		z = torch.cat((z,y.float(), rot.float()), dim=1)
		z = self.latent2hidden(z)
		seq_pred = self.seq_decoder(z)
		z = z.reshape((z.shape[0], 4, -1))

		x = self.decoder_net(z)
		root = self.root_traj(z)
		return x, kld, root, seq_pred



	def sample_gaussian(self, m, v):
		"""
		Element-wise application reparameterization trick to sample from Gaussian
		Args:
			m: tensor: (batch, ...): Mean
			v: tensor: (batch, ...): Variance
		Return:
			z: tensor: (batch, ...): Samples
		"""
		sample = torch.randn(m.shape).to(m.device)
		

		z = m + (v**0.5)*sample
		return z



	def gaussian_parameters(self, h, dim=-1):
		"""
		Converts generic real-valued representations into mean and variance
		parameters of a Gaussian distribution
		Args:
			h: tensor: (batch, ..., dim, ...): Arbitrary tensor
			dim: int: (): Dimension along which to split the tensor for mean and
				variance
		Returns:z
			m: tensor: (batch, ..., dim / 2, ...): Mean
			v: tensor: (batch, ..., dim / 2, ...): Variance
		"""
		m, h = torch.split(h, h.size(dim) // 2, dim=dim)
		v = F.softplus(h) + 1e-8
		return m, v



	def log_normal(self, x, m, v):
		"""
		Computes the elem-wise log probability of a Gaussian and then sum over the
		last dim. Basically we're assuming all dims are batch dims except for the
		last dim.
		Args:
			x: tensor: (batch, ..., dim): Observation
			m: tensor: (batch, ..., dim): Mean
			v: tensor: (batch, ..., dim): Variance
		Return:
			kl: tensor: (batch1, batch2, ...): log probability of each sample. Note
				that the summation dimension (dim=-1) is not kept
		"""

		const = -0.5*x.size(-1)*torch.log(2*torch.tensor(np.pi))
		log_det = -0.5*torch.sum(torch.log(v), dim = -1)
		log_exp = -0.5*torch.sum( (x - m)**2/v, dim = -1)
		log_prob = const + log_det + log_exp

		return log_prob


	def log_normal_mixture(self, z, m, v):
		"""
		Computes log probability of a uniformly-weighted Gaussian mixture.
		Args:
			z: tensor: (batch, dim): Observations
			m: tensor: (batch, mix, dim): Mixture means
			v: tensor: (batch, mix, dim): Mixture variances
		Return:
			log_prob: tensor: (batch,): log probability of each sample
		"""
		z = z.unsqueeze(1)
		log_probs = self.log_normal(z, m, v)
		log_prob = self.log_mean_exp(log_probs, 1)

		return log_prob

	def log_mean_exp(self, x, dim):
		"""
		Compute the log(mean(exp(x), dim)) in a numerically stable manner
		Args:
			x: tensor: (...): Arbitrary tensor
			dim: int: (): Dimension along which mean is computed
		Return:
			_: tensor: (...): log(mean(exp(x), dim))
		"""
		return self.log_sum_exp(x, dim) - np.log(x.size(dim))


	def log_sum_exp(self, x, dim=0):
		"""
		Compute the log(sum(exp(x), dim)) in a numerically stable manner
		Args:
			x: tensor: (...): Arbitrary tensor
			dim: int: (): Dimension along which sum is computed
		Return:
			_: tensor: (...): log(sum(exp(x), dim))
		"""
		max_x = torch.max(x, dim)[0]
		new_x = x - max_x.unsqueeze(dim).expand_as(x)
		return max_x + (new_x.exp().sum(dim)).log()
