import numpy as np
import h5py as h5
import os
from scipy.spatial.transform import Rotation as R
import torch


parent_array = [19,18,1,2,3,1,5,6,8,8,9,10,8,12,13, 11, 14, 8, 17, 1, 19, 19, 19, 21, 22]
datadir = '../dataset/data/'

def root_relative_to_view_norm(poses):
	"""
	    We only want to rotation about y-axis co-ordinate system
	    hence we find the rotation matrix
	"""
	r = poses[:,4,:]
	l = poses[:,8,:]
	y = np.tile(np.array([0., 1., 0.]),(poses.shape[0],1))
	x = (l - r) * np.array([1., 0., 1.])
	x = x / np.linalg.norm(x,keepdims=True)
	z = np.cross(x, y)
	transform_mats = np.stack([x, y, z], axis=1)
	return transform_mats


def rotation_matrix_from_vectors(a, b):
	""" Find the rotation matrix that aligns vec1 to vec2
	:param vec1: A 3d "source" vector
	:param vec2: A 3d "destination" vector
	:return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
	"""

	if not all(a==0):
		a = a/np.linalg.norm(a,keepdims=True)
	b = b/np.linalg.norm(b,keepdims=True)
	s = 0
	v = np.cross(a, b)
	c = np.dot(a, b)
	if not all(v==0):
		s = np.linalg.norm(v,ord=2)

	if s == 0:
		return np.eye(3)
	else:
		kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
		rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
		return rotation_matrix




def create_mask(data_len, T):
	data_len = torch.tensor(data_len)
	max_len = data_len.data.max()
	batch_size = data_len.shape[0]
	seq_range = torch.arange(0,T).long()

	seq_range_ex = seq_range.unsqueeze(0).expand(batch_size, T)
	seq_range_ex = seq_range_ex.unsqueeze(2)
	seq_range_ex = seq_range_ex.expand(batch_size, T, 72)

	seq_len = data_len.unsqueeze(1).expand(batch_size, T)
	seq_len = seq_len.unsqueeze(2)
	seq_len = seq_len.expand(batch_size, T, 72)

	return seq_range_ex < seq_len



def local2matrix(X):
    N,J,D = X.shape
    angle_rotation_matrix = np.zeros((N,J,3,3))
    def dfs(p, rot_param_list):
        print("Completeing:",p)
        for c,x in enumerate(parent_array):
            if x == p and p != c:
                translation_vector = np.zeros((N,3))
                for i in range(N):
                    if i % 5000 == 0:
                        print("Done:{}/{} joint: {}".format(i,N,p))
                    parent_pose_rel = X[i,p,:]
                    child_pose_rel = X[i,c,:]
                    for rot_mat,trans_vec in rot_param_list:
                        parent_pose_rel = (parent_pose_rel - trans_vec[i,:]).dot(rot_mat[i,:,:].T) + trans_vec[i,:]
                        child_pose_rel = (child_pose_rel - trans_vec[i,:]).dot(rot_mat[i,:,:].T) + trans_vec[i,:]
                    rot_mat = rotation_matrix_from_vectors(child_pose_rel-parent_pose_rel, np.array([0,0,1.0]))
                    angle_rotation_matrix[i,c,:,:] = rot_mat
                    translation_vector[i,:] = parent_pose_rel
                dfs(c,rot_param_list + [(angle_rotation_matrix[:,c,:,:],translation_vector)])
    dfs(8,[])
    return angle_rotation_matrix



def get_data_len(x, T):
	data_len = []
	
	for i in range(x.shape[0]):
		t = T-1
		if np.sum(x[i,T-1,:,:]) == np.sum(x[i,T-2,:,:]):
			for t in range(T-1, 0, -1):
				if np.sum(x[i,t,:,:]) != np.sum(x[i,t-1,:,:]):
					break
		data_len.append(t)
	return np.array(data_len)


def transform_X(x):
	T = 256
	x = x.reshape((-1 ,T, 2, 24, 3))
	# x = x[:, :, 0,:,:]
	# x = x[:,:,:,:] - x[:,:,8:9,:]
	print(x.shape)
	N,J,P,D,_ = x.shape

	data_len1 = get_data_len(x[:,:,0,:,:], T)
	data_len2 = get_data_len(x[:,:,1,:,:], T)
	print(data_len1, data_len2)
	mask1 = create_mask(data_len1, T)
	mask2 = create_mask(data_len2, T)
	mask = np.zeros((N,J,P,D*3))
	mask[:,:,0,:] = mask1
	mask[:,:,1,:] = mask2
	mask = mask.reshape((N, J, -1))
	print(mask.shape)

	# x = x.reshape((N*J, D, 3))
	# # Calculating bone length
	# bonelength = np.linalg.norm(x[:, :, :] - x[:, parent_array, :],axis=2,ord=2)
	# idx = bonelength != 0
	# bone = np.sum(bonelength, axis=0) / (np.sum(idx, axis=0) + 10e-17)
	# print('Bone length:', bone.shape)	

	# plot(x.reshape((-1, 300, 25, 3)))
	# rot = local2matrix(x)

	# # Rotation Matrix to Quaternion
	# rotmat = rot.reshape((-1, 3,3))
	# r = R.from_matrix(np.array(rotmat))
	# r = r.as_quat()
	# q = r.reshape((N,J,-1))
	# print('Data length: ', q.shape)
	
	return mask


def distance(a,b):
	return np.sqrt(np.sum((a-b)**2))


def get_root(bbox, root):

	root_list = []
	for i in range(bbox.shape[0]):
		b1 = bbox[i,0,:4]
		b2 = bbox[i,0,4:]

		b1 = np.array([(b1[0] + b1[2])/2, (b1[1] + b1[3])/2])
		b2 = np.array([(b2[0] + b2[2])/2, (b2[1] + b2[3])/2])

		d1 = distance(b1, [0,0])
		d2 = distance(b2, [0,0])
		
		r1 = distance(root[i,0,0,:], [0,0,0])
		r2 = distance(root[i,0,1,:], [0,0,0])

		if r1 >= r2:
			right = root[i,:,0,:]
			left = root[i,:,1,:]
		else:
			right = root[i,:,1,:]
			left = root[i,:,0,:]

		if d1 <= d2:
			left, right = right, left
		# print(left.shape, right.shape)
		r = np.zeros((256,2,3))
		r[:,0,:] = root[i,:,0,:]
		r[:,1,:] = root[i,:,1,:]

		root_list.append(r)
	return np.array(root_list)




if __name__ == "__main__":
	path = os.path.join(datadir, 'NTU_VIBE_CSet_120.h5')
	f = h5.File(path, 'r')

	train_X = f['x'][:]
	# train_bbox = f['bbox'][:]
	# train_y = np.argmax(f['y'][:],-1)

	# test_X = f['test_x'][:]
	# test_bbox = f['test_bbox'][:]
	# test_Y = np.argmax(f['test_y'][:], -1)
	print('Processing Training data')
	train_mask = transform_X(train_X)
	# print('Processing Test data')
	# test_mask = transform_X(test_X)

	print("Saving Data...")
	with h5.File(os.path.join(datadir, 'NTU_mask.h5'), 'w') as f:
		f.create_dataset('train_mask', data=train_mask)
		# f.create_dataset('test_mask', data=test_mask)
	print("Mask saved")

	# print('Root processing started...')
	# train_root = np.load(os.path.join(datadir, 'Train_root.npy'))
	# train_root = get_root(train_bbox, train_root)
	# print(train_root.shape)
	# np.save(os.path.join(datadir, 'Train_root_ordered.npy'), train_root)

	# test_root = np.load(os.path.join(datadir, 'Test_root.npy'))
	# test_root = get_root(test_bbox, test_root)
	# print(test_root.shape)
	# np.save(os.path.join(datadir, 'Test_root_ordered.npy'), test_root)


	
	