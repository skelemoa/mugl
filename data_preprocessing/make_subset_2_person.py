import numpy as np
import h5py
import random
import os


# classes = ["drinking water", "jump up", "make phone call", "hand waving",
# 			"standing up", "wear jacket", "sitting down", "throw", 
# 			"cross hand in front","kicking something"]


# cls_id = [1, 27, 28, 23, 9, 14, 8, 94, 40, 24]
# cls_id = [1,2,3,4,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,54,56,57,61,74]
# cls_id = [i-1 for i in cls_id]


# classes = ["kicking something", "kicking", "butt kicks"]
# cls_id = [24, 51, 100]
# cls_id = [i-1 for i in cls_id]

# classes = ["kicking something", "walking towards", "walking apart"]
# cls_id = [24, 59, 60]
# cls_id = [i-1 for i in cls_id]

dir = '../dataset/data/'

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
		  "kicking something", "kicking", "walking towards", "walking apart",] # for ntu-60


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

# cls_id = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,18, 15, 16, 26, 43, 52, 17, 19,20,21,22,23,25,27,28, 29,
# 			 30,31,32,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50, 53,54,55,56,57,58,
# 			 24, 51, 59, 60] # for ntu-60
cls_id = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,18, 15, 16, 26, 43, 52, 17, 19,20,21,22,23,25,27,28, 29,
			 30,31,32,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50, 53,54,55,56,57,58,24, 51, 59, 60,
			 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 
			88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
			111, 112, 113, 114, 115, 116, 117, 118, 119, 120] # for NTU-120


cls_id = [i-1 for i in cls_id]

print(len(classes), len(cls_id))

# leg_mv_cls = [24, 51, 59, 60] # for ntu-60
leg_mv_cls = [24, 51, 59, 60, 100, 102, 99] # for NTU-120
leg_mv_cls = [i-1 for i in leg_mv_cls]

# cls_id = np.arange(60)

def select_data(x,y, rot6d, mask, root, seq, oversample=50):
	data = []
	label = []
	mean_list = []
	rot6d_list = []
	mask_list = []
	cam_list = []
	root_list = []
	seq_list = []
	for p,i in enumerate(cls_id):
		idx = np.where(y==i)
		x1 = x[idx]
		y1 = y[idx]
		# mean1 = mean_pose[idx]
		rot6d1 = rot6d[idx]
		mask1 = mask[idx]
		root1 = root[idx]
		seq1 = seq[idx]
		# cam1 = camera[idx]
		print(x1.shape, p)
		for j in range(x1.shape[0]):
			if i in leg_mv_cls:
				for OS in range(oversample):
					# data.append(x1[j])
					label.append(p)
					# mean_list.append(mean1[j])
					rot6d_list.append(rot6d1[j])
					mask_list.append(mask1[j])
					root_list.append(root1[j])
					seq_list.append(seq1[j])
			else:
				# data.append(x1[j])
				label.append(p)
				# mean_list.append(mean1[j])
				rot6d_list.append(rot6d1[j])
				mask_list.append(mask1[j])
				root_list.append(root1[j])
				seq_list.append(seq1[j])
			# cam_list.append(cam1[j])
	# data = np.array(data)
	label = np.array(label)
	print('label array created')
	# mean_list = np.array(mean_list)
	rot6d_list = np.array(rot6d_list)
	print('Rotation array created')
	mask_list = np.array(mask_list)
	root_list = np.array(root_list)
	seq_list = np.array(seq_list)
	# cam_list = np.array(cam_list)

	y = np.zeros((label.shape[0], 120))
	y[np.arange(label.shape[0]), label] = 1
	index = np.arange(y.shape[0])
	random.shuffle(index)
	print(rot6d_list.shape, y.shape)
	# data = data[index,:,:]
	# y = y[index,:]
	# # mean_list = mean_list[index]
	# rot6d_list = rot6d_list[index]
	# print('done..')
	# mask_list = mask_list[index]
	# root_list = root_list[index]
	# seq_list = seq_list[index]
	# cam_list = cam_list[index]


	# data = data.reshape(((-1 ,data.shape[1], 2, 24, 3)))
	# data = data[:, :, 0,:,:]
	# data = data.reshape((data.shape[0], data.shape[1], -1))

	# mean_list = mean_list.reshape(((-1 ,mean_list.shape[1], 2, 24, 3)))
	# mean_list = mean_list[:, :, 0,:,:]
	# mean_list = mean_list.reshape((mean_list.shape[0], mean_list.shape[1], -1))


	## selecting alternative frames and cyclic padding
	# data = data[:,::2,:]
	# mean_list = mean_list[:,::2,:]
	# rot6d_list = rot6d_list[:,::2,:]
	# mask_list = mask_list[:,::2,:]
	# mask_list = mask_list[:,:64,:]

	# N,T,_ = data.shape
	# for i in range(N):
	# 	t = T-1
	# 	if np.sum(data[i,T-1,:]) == np.sum(data[i,T-2,:]):
	# 		for t in range(T-1,0,-1):
	# 			if np.sum(data[i,t,:]) != np.sum(data[i,t-1,:]):
	# 				data[i,t:,:] = data[i,:(T-t),:]
	# 				mean_list[i,t:,:] = mean_list[i,:(T-t),:]
	# 				rot6d_list[i,t:,:] = rot6d_list[i,:(T-t),:]
	# 				mask_list[i,t:,:] = mask_list[i,:(T-t),:]


	print(y.shape, rot6d_list.shape, mask_list.shape, root_list.shape, seq_list.shape)
	return y, rot6d_list, mask_list, root_list, seq_list




if __name__ == '__main__':
	f = h5py.File(os.path.join(dir, 'NTU_VIBE_CSet_120.h5'), 'r')
	f1 = h5py.File(os.path.join(dir, 'NTU_mask.h5'), 'r')
	f2 = h5py.File(os.path.join(dir, 'Sequence_120.h5'), 'r')
	x = f['x'][:]
	y = f['y'][:] -1 
	# mean_pose = f['mean_pose'][:]
	rot6d = f['rot6d'][:]
	# camera = f['camera'][:]
	# euler = f['euler'][:]
	mask = f1['train_mask']
	seq = f2['seq'][:]
	train_root = np.load(os.path.join(dir,'Train_root.npy'))

	mask = mask[:,:256,:]
	x = x[:,::4,:]
	rot6d = rot6d[:,::4,:]
	# mean_pose = mean_pose[:,::4,:]
	mask = mask[:,::4,:]
	train_root = train_root[:,::4,:]
	print(x.shape,y.shape, rot6d.shape, mask.shape, seq.shape)


	# test_x = f['test_x'][:]
	# test_y = f['test_y'][:]-1
	# # # test_mean_pose = f['test_mean_pose'][:]
	# test_rot6d = f['test_rot6d'][:]
	# # # test_camera = f['test_camera'][:]
	# # # test_euler = f['test_euler'][:]
	# test_mask = f1['test_mask']
	# test_seq = f2['test_seq'][:]
	# test_root = np.load(os.path.join(dir,'Test_root.npy'))

	# test_mask = test_mask[:,:256,:]
	# test_x = test_x[:,::4,:]
	# test_rot6d = test_rot6d[:,::4,:]
	# # test_mean_pose = test_mean_pose[:,::4,:]
	# test_mask = test_mask[:,::4,:]
	# test_root = test_root[:,::4,:]


	# test_x = test_x[15000: 25000]
	# test_y = test_y[15000: 25000]
	# test_rot6d = test_rot6d[15000: 25000]
	# test_mask = test_mask[15000: 25000]
	# test_root = test_root[15000: 25000]


	y,rot6d,mask,train_root, train_seq = select_data(x,y, rot6d, mask, train_root, seq)
	# test_x,test_y,test_rot6d,test_mask,test_root, test_seq = select_data(test_x,test_y, test_rot6d, test_mask,test_root, test_seq,oversample=1)

	f = h5py.File(os.path.join(dir,'NTU_oversample_120_256.h5'), 'w')
	# f.create_dataset('x', data=x)
	# f.create_dataset('test_x', data=test_x)
	f.create_dataset('y', data=y)
	# f.create_dataset('test_y', data=test_y)
	# f.create_dataset('mean_pose', data=mean_pose)
	# f.create_dataset('test_mean_pose', data=test_mean_pose)
	f.create_dataset('rot6d', data=rot6d)
	# f.create_dataset('test_rot6d', data=test_rot6d)
	# f.create_dataset('camera', data=camera)
	# f.create_dataset('test_camera', data=test_camera)
	f.create_dataset('mask', data=mask)
	# f.create_dataset('test_mask', data=test_mask)
	f.create_dataset('root', data=train_root)
	# f.create_dataset('test_root', data=test_root)
	f.create_dataset('seq', data=train_seq)
	# f.create_dataset('test_seq', data=test_seq)
	print('Data created')




