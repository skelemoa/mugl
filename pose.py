import os
import sys
import h5py
import numpy as np
import traceback  
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from rotation.rotation import rot6d_to_rotmat, batch_rigid_transform
from model.model import *
from utils.util import *
import math

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


classes1 = ["drink water","eat meal","brush teeth","brush hair","drop",
		  "pick up", "throw",	"sit down", "stand up", "clapping",	"reading",
		  "writing", "tear up paper", "put on jacket", "put on glasses", "take off jacket", "put on a shoe", "hopping", "falling down", "pushing",
		  "take off a shoe", "take off glasses", "put on a hat_cap",
		  "take off a hat_cap",	"cheer up",	"hand waving", "reach into pocket", "jump up", "phone call", "play with phone_tablet",
		  "type on a keyboard", "point to something", "taking a selfie",
		  "check time (from watch)", "rub two hands", "nod head_bow", "shake head",
		  "wipe face", "salute", "put palms together", "cross hands in front",
		  "sneeze_cough", "staggering", "headache",
		  "chest pain", "back pain", "neck pain", "nausea_vomiting",
		  "fan self", "punch_slap", "pat on back",
		  "point finger", "hugging", "giving object", "touch pocket",
		  "shaking hands",
		  "kicking something", "kicking", "walking towards", "walking apart"] # for NTU-60

classes2 = ["put on headphone","take off headphone","shoot at basket","bounce ball",
		  "tennis bat swing","juggle table tennis ball","hush","flick hair",
		  "thumb up","thumb down","make OK sign","make victory sign",
		  "staple book","counting money","cutting nails","cutting paper",
		  "snap fingers","open bottle","sniff_smell","squat down",
		  "toss a coin","fold paper","ball up paper","play magic cube",
		  "apply cream on face","apply cream on hand","put on bag","take off bag",
		  "put object into bag","take object out of  bag","open a box","move heavy objects",
		  "shake fist","throw up cap_hat","capitulate","cross arms",
		  "arm circles","arm swings","run on the spot","butt kicks","cross toe touch","side kick",
		  "yawn","stretch oneself","blow nose","hit with object",
		  "wield knife","knock over","grab stuff","shoot with gun",
		  "step on foot","high-five","cheers and drink","carry object",
		  "take a photo","follow","whisper","exchange things",
		  "support somebody","rock-paper-scissors"] # for NTU-61-120

classes = classes1 + classes2

# leg_cls = [56,57,58,59] # for NTU-60
leg_cls = [56,57,58,59,99,98,101] # for NTU-120

# SMPL = np.load("./files/SMPLX_NEUTRAL.npz")
# parent_array = SMPL['kintree_table'][0][:24]
skeleton = np.load('./files/skeleton.npy')
parent_array = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15])

main_path = "./dataset/data/"
learning_rate = 1.5e-2
batch_size = 100
max_epochs = 200
H_size = 100
latent_dim = 352
num_workers = 10
lambda1 = 10
lambda2 = 0.1
device = torch.device('cuda:0')
load_epoch = -1
num_class = 120


def train(epoch, model, loader, optim, L1Loss, L):
	global lambda2
	total_loss = 0
	loss_3d = 0
	loss_6d = 0
	loss_kld = 0
	loss_seq = 0
	total_root_loss = 0
	# if epoch > 0:
	# 	lambda2 = 0.1
	# if epoch > 130:
	# 	lambda2 = 0.1
	for i, (rot6d, mask, y, root, seq) in enumerate(loader):
		lambda2 = L[i] # cyclic annealing schedule
		# skip = 4
		# x = x[:,::skip,:]
		# rot6d = rot6d[:,::skip,:]
		# mask = mask[:,::skip,:]
		leg_mask = torch.zeros(y.shape)
		for idx, p in enumerate(y):
			if p in leg_cls:
				leg_mask[idx] = 1

		label = np.zeros((y.shape[0], num_class))
		label[np.arange(y.shape[0]), y] = 1
		label = torch.tensor(label)


		optim.zero_grad()
		rot = rot6d[:,0,:].reshape((rot6d.shape[0], 48, 6))
		rot = rot[:,0,:]

		pred, kld, root_pred, seq_pred = model(rot6d.to(device), label.to(device), rot.to(device), root.to(device), seq.to(device).float())
		pred_3d1 = fkt(pred[:,:,:144].contiguous(), skeleton, device, parent_array)
		pred_3d2 = fkt(pred[:,:,144:].contiguous(), skeleton, device, parent_array)

		x1 = fkt(rot6d[:,:,:144].to(device).contiguous(), skeleton, device, parent_array)
		x2 = fkt(rot6d[:,:,144:].to(device).contiguous(), skeleton, device, parent_array)
		x = torch.zeros((x1.shape[0], x1.shape[1], x1.shape[2]*2))
		x[:,:,:72] = x1
		x[:,:,72:] = x2

		# maeloss_3d = L1Loss(pred_3d, x.to(device))
		# maeloss_6d = L1Loss(pred, rot6d.to(device))
		seq_loss = L1Loss(seq.to(device).float(), seq_pred)
		maeloss_3d, maeloss_6d, leg_6d, root_loss = loss_function(x, pred, mask, pred_3d1, pred_3d2, rot6d, leg_mask.to(device), root.to(device), root_pred, device) 
		# Experiment 2: leg loss hyper-parameter: 30
		loss = 10*(maeloss_3d + lambda1*maeloss_6d + 30*leg_6d) + lambda2*kld + root_loss + 2*seq_loss
		loss.backward()
		optim.step()
		total_loss += loss.cpu().data.numpy()*x.shape[0]
		loss_3d += maeloss_3d.cpu().data.numpy()*x.shape[0]
		loss_6d += maeloss_6d.cpu().data.numpy()*x.shape[0]
		loss_kld += kld.cpu().data.numpy()*x.shape[0]
		total_root_loss += root_loss.cpu().data.numpy()*x.shape[0]
		loss_seq += seq_loss.cpu().data.numpy()*x.shape[0]

	total_loss /= len(loader.dataset)
	loss_3d /= len(loader.dataset)
	loss_6d /= len(loader.dataset)
	loss_kld /= len(loader.dataset)
	total_root_loss /= len(loader.dataset)
	loss_seq /= len(loader.dataset)

	return total_loss, loss_3d, loss_kld, total_root_loss, loss_seq


def infer(model, epoch, rot, label):
	model.eval()
	# z = torch.randn(6, latent_dim).to(device).float()
	
	# y = np.repeat(np.arange(3),2)
	y = np.arange(num_class)
	rot_list = []
	for i in y:
		idx = np.where(label==i)
		rot_lbl = rot[idx]
		rand = np.random.randint(rot_lbl.shape[0])
		rot_list.append(rot_lbl[rand])
	rot = np.array(rot_list)
	# rot = torch.tensor(rot[:,0,:]).to(device).float()
	rot = rot[:,0,:].reshape((rot.shape[0],48,6))
	rot = rot[:,0,:]
	rot = torch.tensor(rot).to(device).float()

	label = np.zeros((y.shape[0], num_class))
	label[np.arange(y.shape[0]), y] = 1
	label = torch.tensor(label).to(device).float()
	with torch.no_grad():
		m, v = model.gaussian_parameters(model.z_pre.squeeze(0), dim=0)
		idx = torch.distributions.categorical.Categorical(model.pi).sample((label.shape[0],))
		m, v = m[idx], v[idx]
		z = model.sample_gaussian(m, v)
		
		z = torch.cat((z,label,rot), dim=1)
		z = model.latent2hidden(z)
		z = z.reshape((z.shape[0], 4, -1))
		pred = model.decoder_net(z)
		root_pred = model.root_traj(z).unsqueeze(2)
		pred_3d1 = fkt(pred[:,:,:144].contiguous(), skeleton, device, parent_array)
		pred_3d2 = fkt(pred[:,:,144:].contiguous(), skeleton, device, parent_array)
		# pred_3d = fkt(pred, skeleton).cpu().data.numpy()
		pred_3d1 = pred_3d1.reshape((pred_3d1.shape[0], pred_3d1.shape[1], 24,-1)).cpu().data.numpy()
		pred_3d2 = pred_3d2.reshape((pred_3d2.shape[0], pred_3d2.shape[1], 24,-1)).cpu().data.numpy()
		root_pred = root_pred.cpu().data.numpy()
		pred_3d2 = pred_3d2 + root_pred

		plot_infer(epoch,y, y, pred_3d1,pred_3d2)



if __name__ == '__main__':
	train_loader, N = get_datasets(main_path, batch_size, num_workers)

	model = Model(num_class, latent_dim).to(device)

	total_params = sum(p.numel() for p in model.parameters())
	print('Total number of parameters:', total_params)

	optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.001)
	scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
	L1Loss = nn.L1Loss().to(device)

	train_loss_list = []
	test_loss_list = []

	# Cyclic Annealing
	no_itr = math.ceil(N/batch_size)
	L = frange_cycle_linear(no_itr)
	# print(L)
	if load_epoch > 0:
		model.load_state_dict(torch.load('./checkpoints/' + 'model_{}.pt'.format(load_epoch), map_location=torch.device('cpu')))


	for epoch in range(load_epoch+1, max_epochs):
		model.train()
		train_loss, train_recon, train_kld, train_root, train_seq = train(epoch, model, train_loader, optimizer, L1Loss, L)
		# with torch.no_grad():
		# 	model.eval()
		# 	test_loss, test_recon, test_kld, test_seq, total_mmd = test(epoch, model, val_loader, L1Loss)
		# 	if epoch>150:
		# 		infer(model, epoch, test_rot6d, test_y)
		for param_group in optimizer.param_groups:
			print('Learning Rate:',param_group['lr'])
		print('Epoch: {}/{} Train Loss:{} Recon: {} kld:{} Root: {} Seq: {}'.format(epoch, max_epochs, train_loss, train_recon, train_kld, train_root, train_seq))
		# print('Epoch: {}/{} Test Loss:{} Recon: {} kld: {}'.format(epoch, max_epochs, test_loss, test_recon, test_kld))
		print('======================================================================================')

		train_loss_list.append([train_loss, train_recon, train_kld])
		# test_loss_list.append([test_loss, test_recon, test_kld])
		if not os.path.isdir('./loss'):
			os.mkdir('./loss')

		np.save('./loss/Train_loss', np.array(train_loss_list))
		# np.save('./loss/Test_loss', np.array(test_loss_list))

		if epoch > 20:
			scheduler.step()

		if epoch > 100:
			save_model(model,epoch)





