import os
import h5py
import numpy as np  
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.utils.data import Dataset, DataLoader
from rotation.rotation import rot6d_to_rotmat, batch_rigid_transform


class NTUDataset(Dataset):
    def __init__(self,rot6d,mask, y, root, seq, mean_pose):
        # self.x = x[:,:,:]
        self.rot6d = rot6d[:,:,:]
        self.y = y
        self.mask = mask[:,:,:]
        self.root = root[:,:,:,:]
        self.mean_pose = mean_pose
        self.res1 = self.root[:,:,1,:] - self.root[:,:,0,:]
        self.res2 = self.root[:,:,0,:] - np.zeros((self.root[:,:,0,:].shape))

        self.residual = np.zeros((self.root.shape[0], self.root.shape[1], 2*3))
        self.residual[:,:,:3] = self.res2
        self.residual[:,:,3:] = self.res1
        print('Residual shape:', self.res1.shape, self.res2.shape, self.residual.shape)

        self.N = self.rot6d.shape[0]

        n,t = seq.shape
        self.seq = seq.reshape((n,t,-1))
        print(self.rot6d.shape, self.seq.shape)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return [self.rot6d[index], self.mask[index], self.y[index], self.residual[index], self.seq[index], self.mean_pose[index]]



def reconstruction(y, epoch,pred, gr,root,loader,visualize_data=False):
	if pred.shape != gr.shape:
		print("Prediction and Ground truth not of the same shape. Pred:{} Gr:{}".format(pred.shape,gr.shape))
		return None
	N,T,D = pred.shape
	pred = pred.reshape((N,T,D//3,3))
	gr = gr.reshape((N,T,D//3,3))
	y = y.cpu().data.numpy()
	gr = gr + root
	# print(gr.shape, root.shape)

	# if visualize_data and epoch > 150:
	# 	plot(y, epoch,gr,pred)

	
def fkt(x, mean_pose, device, parent_array):
	# forward kinematics
	rotmat = rot6d_to_rotmat(x)
	# same mean pose across timesteps
	mean_pose = torch.tensor(mean_pose.reshape((x.shape[0],1, -1)))
	mean_pose = mean_pose.expand((x.shape[0], x.shape[1], 72))
	mean_pose = mean_pose[:,:,:].reshape((x.shape[0]*x.shape[1],-1,3))
	rotmat = rotmat.reshape((x.shape[0]*x.shape[1],-1, 3, 3))
	pred = batch_rigid_transform(rotmat.float(),mean_pose.to(device).float(),parent_array)
	x = pred.reshape((x.shape[0], x.shape[1], -1))
	return x


def loss_function(x, pred, mask, pred_3d1, pred_3d2, rot6d, leg_mask, root, root_pred, device):
	leg = [1,2,4,5,7,8,10,11, 25, 26, 28, 29, 31, 32, 34, 35]
	maeloss_3d1 = torch.sum(torch.mul(torch.abs(pred_3d1[:,:,:] - x[:,:,:72].to(device)), mask[:,:,:72].to(device).float())) / (torch.sum(mask[:,:,:72].to(device).float()) + 10e-8)
	maeloss_3d2 = torch.sum(torch.mul(torch.abs(pred_3d2[:,:,:] - x[:,:,72:].to(device)), mask[:,:,72:].to(device).float())) / (torch.sum(mask[:,:,72:].to(device).float()) + 10e-8)
	maeloss_3d = maeloss_3d1 + maeloss_3d2


	mask1 = mask.unsqueeze(3).expand(mask.shape[0], mask.shape[1], mask.shape[2], 2).reshape((mask.shape[0], mask.shape[1], -1))
	maeloss_6d = torch.sum(torch.mul(torch.abs(pred[:,:,:] - rot6d[:,:,:].to(device)), mask1[:,:,:].to(device).float())) / (torch.sum(mask1[:,:,:].to(device).float()) + 10e-8)
	
	rot6d = rot6d.reshape((rot6d.shape[0], rot6d.shape[1], 48, 6))
	pred = pred.reshape((pred.shape[0], pred.shape[1], 48, 6))
	mask1 = mask1.reshape((mask1.shape[0], mask1.shape[1], 48, 6))

	l = torch.sum(torch.mul(torch.abs(pred[:,:,leg,:] - rot6d[:,:,leg, :].to(device)), mask1[:,:,leg,:].to(device).float()), dim=(1,2,3))
	leg_6d = l / (torch.sum(mask1[:,:,leg,:].to(device).float(), dim=(1,2,3)) + 10e-8)
	leg_6d = torch.mean(leg_6d*leg_mask)


	# root = root[:,:,1,:] - root[:,:,0,:]
	root_mask = mask[:,:,:6]
	root_loss = torch.sum(torch.mul(torch.abs(root.float() - root_pred), root_mask.to(device).float()))/(torch.sum(root_mask.to(device).float()) + 10e-8)

	return maeloss_3d, maeloss_6d, leg_6d, root_loss





def plot_infer(epoch,y1, y2, X,pred):

	if not os.path.isdir("./image_test"):
		os.mkdir("./image_test")

	if not os.path.isdir(os.path.join("./image_test",str(epoch))):
		os.mkdir(os.path.join("./image_test",str(epoch)))

	N,T,J,_ = pred.shape
	for i in range(0,N):
		fig = plt.figure(figsize=(8,4))
		ax = fig.add_subplot(111,projection='3d')
		ax.view_init(azim=-90,elev=-90)
		name1 = classes[y1[i]]
		name2 = classes[y2[i]]
		for j in range(T):
			plt.cla()
			# if np.sum(X[i, j,:,:])==0:
			# 	break
			gr_pose = X[i, j,:,:] - np.mean(X[i, j,:,:],axis=0,keepdims=True)

			pred_pose = pred[i, j,:,:] - np.mean(pred[i, j,:,:],axis=0,keepdims=True)
			# pred_pose[:,0] += 0.6

			ax.scatter(gr_pose[:,0],gr_pose[:,1],gr_pose[:,2],s=100,c="green",label="Person 1")
			ax.scatter(pred_pose[:,0],pred_pose[:,1],pred_pose[:,2],s=100,c="red",label="Person 2")
			for jj,p in enumerate(parent_array):
				if p == -1:
					continue
				ax.plot([gr_pose[jj,0],gr_pose[p,0]],[gr_pose[jj,1],gr_pose[p,1]],[gr_pose[jj,2],gr_pose[p,2]],c="black",linewidth=3.0)
				ax.plot([pred_pose[jj,0],pred_pose[p,0]],[pred_pose[jj,1],pred_pose[p,1]],[pred_pose[jj,2],pred_pose[p,2]],c="black",linewidth=3.0)
			min_lim = np.min(gr_pose)
			max_lim = np.max(gr_pose)
			ax.set_xlim(min_lim, max_lim)
			ax.set_ylim(min_lim, max_lim)
			ax.set_ylim(min_lim, max_lim)
			ax.legend()
			ax.axis('off')
			# plt.draw()
			plt.title(name1)
			plt.savefig("./image_test/" + str(epoch) + '/'  + str(i*1000 + j) +".png")
		plt.close()	
	print("Plotting Complete")



def plot(y, epoch,X,pred):
	if not os.path.isdir("./image"):
		os.mkdir("./image")

	if not os.path.isdir(os.path.join("./image",str(epoch))):
		os.mkdir(os.path.join("./image",str(epoch)))

	N,T,J,_ = pred.shape
	for i in range(0,N,25):
		fig = plt.figure(figsize=(8,4))
		ax = fig.add_subplot(111,projection='3d')
		ax.view_init(azim=-90,elev=-90)
		name = classes[y[i]]
		for j in range(T):
			plt.cla()
			if np.sum(X[i, j,:,:])==0:
				break
			gr_pose = X[i, j,:,:] - np.mean(X[i, j,:,:],axis=0,keepdims=True)

			pred_pose = pred[i, j,:,:] - np.mean(pred[i, j,:,:],axis=0,keepdims=True)
			# pred_pose[:,0] += 0.6

			ax.scatter(gr_pose[:,0],gr_pose[:,1],gr_pose[:,2],s=100,c="green",label="Person 1")
			ax.scatter(pred_pose[:,0],pred_pose[:,1],pred_pose[:,2],s=100,c="red",label="Person 2")
			for jj,p in enumerate(parent_array):
				if p == -1:
					continue
				ax.plot([gr_pose[jj,0],gr_pose[p,0]],[gr_pose[jj,1],gr_pose[p,1]],[gr_pose[jj,2],gr_pose[p,2]],c="black",linewidth=3.0)
				ax.plot([pred_pose[jj,0],pred_pose[p,0]],[pred_pose[jj,1],pred_pose[p,1]],[pred_pose[jj,2],pred_pose[p,2]],c="black",linewidth=3.0)
			ax.legend()
			ax.axis('off')
			# plt.draw()
			plt.title(name)
			plt.savefig("./image/" + str(epoch) + '/'  + str(i) + '_' + str(j) +".png")
		plt.close()	
	print("Plotting Complete")	





def get_datasets(main_path, batch_size, num_workers):
    datadir = os.path.join(main_path,'data')
    path = os.path.join(datadir, 'NTU_oversample_120_256.h5')
    f = h5py.File(path, 'r')

    ### selecting only kicking class
    # x = f['x'][:]
    rot6d = f['rot6d'][:]
    mean_pose = f['mean_pose'][:]
    mask = f['mask'][:]
    y = np.argmax(f['y'][:], axis=1)
    root = f['root'][:]
    seq = f['seq'][:]


    train_dataset = NTUDataset(rot6d,mask, y, root, seq, mean_pose)
    print(f"Train Dataset Loaded {train_dataset.N} samples")
    
    # test_x = f['test_x'][:]
    # test_rot6d = f['test_rot6d'][:]
    # # test_mean_pose = f['test_mean_pose'][:]
    # test_mask = f['test_mask'][:]
    # test_y = np.argmax(f['test_y'][:], axis=1)
    # test_root = f['test_root'][:]

    # test_dataset = NTUDataset(test_x,test_rot6d,test_mask, test_y, test_root)
    # print(f"Test Dataset Loaded {test_dataset.N} samples")

    train_loader = DataLoader(train_dataset,batch_size=batch_size, num_workers=num_workers,shuffle=True)
    # val_loader = DataLoader(test_dataset,batch_size=batch_size, num_workers=num_workers,shuffle=True)
    return train_loader, rot6d.shape[0]


def save_model(model,epoch):
	if not os.path.isdir("./checkpoints"):
		os.mkdir("./checkpoints")
	
	filename = './checkpoints/' + 'model_{}.pt'.format(epoch)
	torch.save(model.state_dict(), filename)



def frange_cycle_linear(n_iter, start=0.0, stop=0.1,  n_cycle=8, ratio=0.6):
    """
    Implemenataion of cyclic annealing. 
    Function borrowed from https://github.com/haofuml/cyclical_annealing
    If we increase the value of ratio then it decreases the weight on KL-digergence loss
    and increases the weight on reconstruction loss, results in better generation quality 
    for leg movement class but poor generation quality of hand movement classes. Best 
    generation quality for the hand movement classes is observed if we remove cyclic annealing
    completely and put lambda2 = 0.1 but then leg movement classes don't work. So, optimal result 
    is obtained with the current setting of cyclic annealing.  
    Args:
       n_iter: number of iteration within an epoch
       start: initial value of the cycle
       stop: final value of the cycle
       n_cycle: number of cycle 
       ratio: lower ratio indicates more weight on KL-divergence loss
       and vice virsa
    Return:
        L: numpy ndarray: holds hyper-parameter of KL-Divergence loss
        for every iteration 
    """
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L