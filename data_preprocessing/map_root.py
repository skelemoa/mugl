import numpy as np
import ipyvolume as ipv
import h5py
import os
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

kinect_dir = '../dataset/kinect/'
dir = '../dataset/data/'

kinect_files = os.listdir(kinect_dir)
missing_file_count = 0

def get_vibe_dir(x):
    x1 = x[16,:] - x[0,:]
    x2 = x[17,:] - x[0,:]
    return np.cross(x1,x2)


def get_kinect_dir(x):
    x1 = x[8,:] - x[0,:]
    x2 = x[4,:] - x[0,:]
    return np.cross(x1, x2)

def get_kinect_peron(i):
    # this function returns Kinect skeleton given the index
    f = i + '.skeleton.npy'
    p = np.zeros((300,2,25,3))
    if f not in kinect_files:
#         print(f)
        pass
    else:
        kp = np.load(os.path.join(kinect_dir, f))
        if kp.shape[0] != 0:
            if kp.shape[0] == 1:
                p[:,0,:,:] = kp[0,:,:,:]
            else:
                p[:,0,:,:] = kp[0,:,:,:]
                p[:,1,:,:] = kp[1,:,:,:]
    return p[:256,:,:,:]


def order_root(kinect_person, vibe):
    vibe = vibe.reshape((256, 2, 24, 3))
#     kinect_person = kinect_person[::4,:,:,:]
    person = vibe[:,:,:]
    left = kinect_person[:,0,0,:].reshape((256,1,3))
    right = kinect_person[:,1,0,:].reshape((256,1,3))

    person1 = person[:,0,:,:] + left
    person2 = person[:,1,:,:] + right
    
    v1 = get_vibe_dir(person[0,0,:,:])
    v2 = get_vibe_dir(person[0,1,:,:])
    v_cross = np.cross(v1, v2)

    k1 = get_kinect_dir(kinect_person[0,0,:,:])
    k2 = get_kinect_dir(kinect_person[0,1,:,:])
    k_cross = np.cross(k1,k2)
    
    dot_prod = np.sum(v_cross*k_cross)
#     print(dot_prod)

    if dot_prod > 0:
        # right direction
        return left, right
    elif dot_prod < 0:
        # Wrong Direction
        return right, left
    else:
        # one person missing
        return left, right 


def get_root(x, y, train_file_names):
    count = 0
    root_list = []
#     person_2_cls = [50,51,52,53,54,55,56,57,58,59,60]
    person_2_cls =  [50,51,52,53,54,55,56,57,58,59,60,106, 107, 108, 109
                     ,110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
    for i in tqdm(range(train_file_names.shape[0])):
        file_name = train_file_names[i]
        if len(file_name) >= 2 and len(file_name) != 20:
            file_name = file_name[0]
        if str(file_name)[0] == '[':
            file_name = file_name[0]

        root = np.zeros((256, 2, 3))
        if y[i] in person_2_cls:
            p = get_kinect_peron(file_name)
            left, right = order_root(p, x[i])
            
            if y[i] == 60:
                root[:,0:1,:] = right
                root[:,1:,:] = left
            else:
                root[:,0:1,:] = left
                root[:,1:,:] = right
        
        root_list.append(root)
    return np.array(root_list)


if __name__ == '__main__':
	f = h5py.File(os.path.join(dir, 'NTU_VIBE_CSet_120.h5'), 'r')

	# train data
	x = f['x'][:]
	y = f['y'][:]
	train_file_names = np.load(os.path.join(dir, 'Train_File_order.npy'), allow_pickle=True)
	# print(x.shape)
	train_root = get_root(x, y, train_file_names)
	print(train_root.shape)
	np.save(dir + 'Train_root.npy', train_root)

	# test data
	# test_x = f['test_x'][:]
	# test_y = f['test_y'][:]
	# test_file_names = np.load(os.path.join(dir, 'Test_File_order.npy'), allow_pickle=True)
	# test_root = get_root(test_x, test_y, test_file_names)
	# print(test_root.shape)
	# np.save(dir + 'Test_root.npy', test_root)