import os 
import sys 
import json
import numpy as np 
import torch 
from torch.nn import functional as F
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


orig_joints = [] # Original location of the joints of the smpl model 


def rotmat_to_rot6d(x):
	return x[...,:2]

def rot6d_to_rotmat(x):
	x = x.view(-1,3,2)

	# Normalize the first vector
	b1 = F.normalize(x[:, :, 0], dim=1, eps=1e-6)

	dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
	# Compute the second vector by finding the orthogonal complement to it
	b2 = F.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-6)

	# Finish building the basis by taking the cross product
	b3 = torch.cross(b1, b2, dim=1)
	rot_mats = torch.stack([b1, b2, b3], dim=-1)

	return rot_mats

def euler2quat(axisang):
	# This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L37
	# axisang N x 3
	axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
	angle = torch.unsqueeze(axisang_norm, -1)
	axisang_normalized = torch.div(axisang, angle)
	angle = angle * 0.5
	v_cos = torch.cos(angle)
	v_sin = torch.sin(angle)
	quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
	return quat

def quat2mat(quat):
	"""
	This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50
	Convert quaternion coefficients to rotation matrix.
	Args:
		quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
	Returns:
		Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
	"""
	norm_quat = quat
	norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
	w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
															 2], norm_quat[:,
																		   3]

	batch_size = quat.size(0)

	w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
	wx, wy, wz = w * x, w * y, w * z
	xy, xz, yz = x * y, x * z, y * z

	rotMat = torch.stack([
		w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
		w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
		w2 - x2 - y2 + z2
	],
						 dim=1).view(batch_size, 3, 3)
	return rotMat


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert 3x4 rotation matrix to 4d quaternion vector
    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201
    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.
    Return:
        Tensor: the rotation in quaternion
    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`
    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
	"""
	Applies a batch of rigid transformations to the joints
	Parameters
	----------
	rot_mats : torch.tensor BxNx3x3
		Tensor of rotation matrices
	joints : torch.tensor BxNx3
		Locations of joints
	parents : torch.tensor BxN
		The kinematic tree of each object
	dtype : torch.dtype, optional:
		The data type of the created tensors, the default is torch.float32
	Returns
	-------
	posed_joints : torch.tensor BxNx3
		The locations of the joints after applying the pose rotations
	rel_transforms : torch.tensor BxNx4x4
		The relative (with respect to the root joint) rigid transformations
		for all the joints
	"""


	def transform_mat(R, t):
		''' Creates a batch of transformation matrices
			Args:
				- R: Bx3x3 array of a batch of rotation matrices
				- t: Bx3x1 array of a batch of translation vectors
			Returns:
				- T: Bx4x4 Transformation matrix
		'''
		# No padding left or right, only add an extra row

		# print(R.shape,t.shape)

		return torch.cat([F.pad(R, [0, 0, 0, 1]),
						  F.pad(t, [0, 0, 0, 1], value=1)], dim=2)



	joints = torch.unsqueeze(joints, dim=-1)

	# print(joints.shape,rot_mats.shape)

	rel_joints = joints.clone()
	rel_joints[:, 1:] -= joints[:, parents[1:]]

	# rel_joints *= 1.50

	transforms_mat = transform_mat(
		rot_mats.reshape(-1, 3, 3),
		rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

	transform_chain = [transforms_mat[:, 0]]
	for i in range(1, parents.shape[0]):
		# Subtract the joint location at the rest pose
		# No need for rotation, since it's identity when at rest
		curr_res = torch.matmul(transform_chain[parents[i]],
								transforms_mat[:, i])
		transform_chain.append(curr_res)

	transforms = torch.stack(transform_chain, dim=1)

	# The last column of the transformations contains the posed joints
	posed_joints = transforms[:, :, :3, 3]

	return posed_joints


def plot(X,pred,parent_array):

	if not os.path.isdir("./image"):
		os.mkdir("./image")

	N,T,J,_ = pred.shape
	for i in range(N):
		fig = plt.figure(figsize=(8,4))
		ax = fig.add_subplot(111,projection='3d')
		ax.view_init(azim=-90,elev=-90)
		for j in range(batch_timesteps[i]):
			plt.cla()
			gr_pose = X[i, j,:,:] - np.mean(X[i, j,:,:],axis=0,keepdims=True)

			pred_pose = pred[i, j,:,:] - np.mean(pred[i, j,:,:],axis=0,keepdims=True)
			# pred_pose[:,0] += 0.5

			ax.scatter(gr_pose[:,0],gr_pose[:,1],gr_pose[:,2],s=200,c="green",label="Ground Truth")
			ax.scatter(pred_pose[:,0],pred_pose[:,1],pred_pose[:,2],s=200,c="red",label="Prediction")
			for jj,p in enumerate(parent_array):
				if p == -1:
					continue
				ax.text(gr_pose[jj,0],gr_pose[jj,1],gr_pose[jj,2],str(jj),c="black")

				ax.plot([gr_pose[jj,0],gr_pose[p,0]],[gr_pose[jj,1],gr_pose[p,1]],[gr_pose[jj,2],gr_pose[p,2]],c="black",linewidth=3.0)
				ax.plot([pred_pose[jj,0],pred_pose[p,0]],[pred_pose[jj,1],pred_pose[p,1]],[pred_pose[jj,2],pred_pose[p,2]],c="black",linewidth=3.0)
			ax.legend()
			ax.axis('off')
			# plt.draw()
			# plt.show()
			plt.savefig("./image/" + '/'  + str(i) + '_' + str(j) +".png")
		# plt.close()	
	print("Plotting Complete")


if __name__ == "__main__":

	SMPL = np.load("SMPLX_NEUTRAL.npz")
	J_regressor = SMPL['J_regressor']
	parents = SMPL['kintree_table'][0][:24]
	parents[0] = -1
	orig_verts = SMPL['v_template']

	max_t = 128
	index = []

	count = 0
	json_path = "/ssd_scratch/cvit/debtanu.gupta/data/"
	json_file = os.listdir(json_path)
	for i in json_file:
		batch_joints3d = []	
		batch_angles = []
		batch_meanpose = []
		batch_timesteps = []
		labels = []
		setup_list = []
		person_idx = []
		seq_len = []
		
		# if i in ['nturgbd_rgb_s001.zip']:
		# 	continue
		print('Processing: ', i)
		json_data = os.listdir(os.path.join(json_path, i))
		for file in json_data:
			setup_id = int(file[1:4])
			y = int(file.split('_')[0][-3:])

			with open(os.path.join(json_path, i, file),"r") as f:
				data = json.load(f)

			person_list = list(data.keys())
			if person_list == []:
				print('Empty File: ', file)
				continue

			for people in data.keys():
				t = min(max_t,len(data[people]['joints3d']))

				angles = torch.empty((max_t,72))	
				joints3d = torch.empty((max_t,49,3))	
				max_pose = torch.empty((max_t,24,3))

				joints3d[:t] = torch.Tensor(data[people]['joints3d'])[:t]
				angles[:t] = torch.Tensor(data[people]['pred_pose'])[:t]

				betas = np.array(data[people]['pred_betas'])
				mean_pose = SMPL['v_template'][:,:,None] + SMPL['shapedirs'][:,:,:10]@betas.T
				mean_pose = mean_pose.transpose((1,0,2))
				mean_pose = SMPL['J_regressor']@mean_pose
				mean_pose = mean_pose.transpose((2,1,0))
				max_pose[:t] = torch.Tensor(mean_pose)[:t,:24]


				batch_meanpose.append(max_pose)
				batch_joints3d.append(joints3d)
				batch_angles.append(angles)
				batch_timesteps.append(t)
				person_idx.append(count)
				labels.append(y)
				setup_list.append(setup_id)
				seq_len.append(t)
				
			if count%1000==0:
				print(count, file, setup_id, y)
			count += 1
		
		print('File precessed: ', count)

		batch_joints3d = torch.stack(batch_joints3d,dim=0)
		batch_angles = torch.stack(batch_angles,dim=0)
		batch_meanpose = torch.stack(batch_meanpose,dim=0)

		B,T,D = batch_angles.shape
		batch_quat = euler2quat(batch_angles.view(-1,3))	
		batch_rot  = quat2mat(batch_quat).view(B*T,-1,3,3)
		rot6d = rotmat_to_rot6d(batch_rot)
		rot6d = rot6d.reshape((B,T,-1,6))


		batch_meanpose  = batch_meanpose.view(B*T,-1,3)
		print(batch_meanpose.shape)

		start = time.time()
		recreated_joints = batch_rigid_transform(batch_rot,batch_meanpose,parents)
		print(f'Time: {time.time() - start}')
		recreated_joints = recreated_joints.view(B,T,-1,3)
		batch_meanpose  = batch_meanpose.view(B,T,-1,3)
		print('Dataset shape: ', batch_joints3d.shape)
		print('Angle shape: ', batch_angles.shape)
		print('Mean pose shape: ', batch_meanpose.shape)
		print('Reconstructed pose shape: ', recreated_joints.shape)
		print('Rot6d shape:', rot6d.shape)
		labels = np.array(labels)
		print('Label shape: ', labels.shape)
		
		if not os.path.isdir("/ssd_scratch/cvit/debtanu.gupta/files"):
			os.mkdir("/ssd_scratch/cvit/debtanu.gupta/files")
		
		print('Saving Data..')
		np.save('/ssd_scratch/cvit/debtanu.gupta/files/X_{}'.format(setup_id), np.array(recreated_joints))
		np.save('/ssd_scratch/cvit/debtanu.gupta/files/Y_{}'.format(setup_id), np.array(labels))
		np.save('/ssd_scratch/cvit/debtanu.gupta/files/Mean_pose_{}'.format(setup_id), np.array(batch_meanpose))
		np.save('/ssd_scratch/cvit/debtanu.gupta/files/Setup_list_{}'.format(setup_id), np.array(setup_list))
		np.save('/ssd_scratch/cvit/debtanu.gupta/files/Persion_idx_{}'.format(setup_id), np.array(person_idx))
		np.save('/ssd_scratch/cvit/debtanu.gupta/files/Seq_len_{}'.format(setup_id), np.array(seq_len))
		np.save('/ssd_scratch/cvit/debtanu.gupta/files/Rot6d_{}'.format(setup_id), np.array(rot6d))
		index.append(setup_id)
		np.save('/ssd_scratch/cvit/debtanu.gupta/files/index', np.array(index))
		print('Data Saved')
		print('=================================================')
	# plot(recreated_joints.numpy(),recreated_joints.numpy(),parents)	