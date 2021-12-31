import numpy as np
import h5py
import os

main_path = '../dataset/data/'


def get_data(mask, y):
	N,T,_ = mask.shape
	mask = mask.reshape((N,T,2,24,3))
	mask = mask[:,:,0,:,:]
	seq_len = np.sum(mask[:,:,0,0], axis=1)
	idx1 = np.where(seq_len == 255)
	seq_len = seq_len[:, None]
	seq = np.arange(0, 256)
	seq = seq.reshape((1, 256))
	seq = np.repeat(seq, mask.shape[0], axis=0)
	seq = seq/seq_len
	idx = np.where(seq>=1)
	seq[idx] = 1.0
	seq = seq[:,::4]
	seq[:,-1] = 1.0


	return seq, y, seq_len



if __name__ == '__main__':
	f = h5py.File(os.path.join(main_path, 'NTU_mask.h5'), 'r')
	train_mask = f['train_mask'][:]
	# test_mask = f['test_mask'][:]
	f.close()

	f = h5py.File(os.path.join(main_path, 'NTU_VIBE_CSet_120.h5'), 'r')
	train_y = f['y'][:]
	# test_y = f['test_y'][:]

	print(train_mask.shape, train_y.shape)
	

	train_seq, train_y, seq_len = get_data(train_mask, train_y)
	# test_seq, test_y, test_seq_len = get_data(test_mask, test_y)


	print(train_seq.shape, train_y.shape, seq_len.shape)

	f = h5py.File(os.path.join(main_path, 'Sequence_120.h5'), 'w')
	f.create_dataset('seq', data = train_seq)
	# f.create_dataset('test_seq', data = test_seq)

	f.create_dataset('y', data = train_y)
	# f.create_dataset('test_y', data = test_y)

	f.create_dataset('len', data = seq_len)
	# f.create_dataset('test_len', data = test_seq_len)

	f.close()