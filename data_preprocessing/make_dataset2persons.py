import numpy as np
import h5py
import sys
import os


data_list = []
labels = []
mean_list = []
setup_list = []
rot6d_list = []
cam_list = []
e_list = []
bbox_list = []
file_list = []

person_2 = []

main_path = '../dataset'
cls_id = np.arange(1, 121)
# person_2_cls = [50,51,52,53,54,55,56,57,58,59,60]
person_2_cls = [50,51,52,53,54,55,56,57,58,59,60,106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
index = np.load(main_path + '/files_256/index.npy')
for jj in index:

    X = np.load(main_path + '/files_256/X_{}.npy'.format(jj))
    Y = np.load(main_path + '/files_256/Y_{}.npy'.format(jj))
    euler = np.load(main_path + '/files_256/Euler_{}.npy'.format(jj))
    person_idx = np.load(main_path + '/files_256/Persion_idx_{}.npy'.format(jj))
    meanpose = np.load(main_path + '/files_256/Mean_pose_{}.npy'.format(jj))
    setup_id = np.load(main_path + '/files_256/Setup_list_{}.npy'.format(jj))
    seq_len = np.load(main_path + '/files_256/Seq_len_{}.npy'.format(jj))
    rot6d = np.load(main_path + '/files_256/Rot6d_{}.npy'.format(jj))
    camera = np.load(main_path + '/files_256/Camera_{}.npy'.format(jj))
    bbox = np.load(main_path + '/files_256/Bbox_{}.npy'.format(jj))
    file = np.load(main_path + '/files_256/File_name_{}.npy'.format(jj))

    X = X.reshape((X.shape[0], X.shape[1], -1))
    meanpose = meanpose.reshape((meanpose.shape[0], meanpose.shape[1], -1))
    rot6d = rot6d.reshape((rot6d.shape[0], rot6d.shape[1], -1))

    print(X.shape, Y.shape, person_idx.shape, meanpose.shape, euler.shape)

    for i in np.unique(person_idx):
        idx = np.where(i==person_idx)
        
        cam_id = camera[idx][0]
        # if cam_id == 2:
        #     continue

        if Y[idx][0] not in cls_id:
            continue
        # print(Y[idx][0])

        if setup_id[idx][0]%2 != 0:
            continue

        if Y[idx][0] not in person_2_cls:
            # single person class
            data = np.zeros((256, 144))
            mean = np.zeros((256, 144))
            rot = np.zeros((256, 144*2))
            b = np.zeros((256, 8))

            if len(idx[0]) != 1:
                # person 1
                idx = idx[0]
                t = seq_len[idx]
                data[:t[0], :72] = X[idx][0,:t[0],:]
                mean[:t[0], :72] = meanpose[idx][0,:t[0],:]
                rot[:t[0], :144] = rot6d[idx][0,:t[0],:]
                b[:t[0], :4] = bbox[idx][0,:t[0]]
                
                # person 2
                data[:t[0], 72:] = X[idx][0,:t[0],:]
                mean[:t[0], 72:] = meanpose[idx][0,:t[0],:]
                rot[:t[0], 144:] = rot6d[idx][0,:t[0],:]
                b[:t[0], 4:] = bbox[idx][0,:t[0]]

            else:
                t = seq_len[idx][0]
                # person 1
                data[:t, :72] = X[idx][:,:t]
                mean[:t, :72] = meanpose[idx][:,:t]
                rot[:t, :144] = rot6d[idx][:,:t]
                b[:t, :4] = bbox[idx][0,:t]

                # person 2
                data[:t, 72:] = X[idx][:,:t]
                mean[:t, 72:] = meanpose[idx][:,:t]
                rot[:t, 144:] = rot6d[idx][:,:t]
                b[:t, 4:] = bbox[idx][0,:t]


            data_list.append(data)
            mean_list.append(mean)
            labels.append(Y[idx][0])
            cam_list.append(camera[idx][0])
            rot6d_list.append(rot)
            setup_list.append(setup_id[idx][0])
            file_list.append(file[idx])
            bbox_list.append(b)
        else:
            # two person class
            t = seq_len[idx]
            data = np.zeros((256, 144))
            mean = np.zeros((256, 144))
            rot = np.zeros((256, 144*2))
            b = np.zeros((256, 8))

            if t.shape[0] != 1:
                data[:t[0], :72] = X[idx][0, :t[0],:] # person 1
                data[:t[1], 72:] = X[idx][1, :t[1],:] # person 2
                mean[:t[0], :72] = meanpose[idx][0, :t[0],:]
                mean[:t[1], 72:] = meanpose[idx][1, :t[1],:]
                rot[:t[0], :144] = rot6d[idx][0, :t[0],:] # for person 1 
                rot[:t[1], 144:] = rot6d[idx][1, :t[1],:] # for person 1
                b[:t[0], :4] = bbox[idx][0,:t[0]]
                b[:t[1], 4:] = bbox[idx][1,:t[1]]
            else:
                # if either of ther persons is missing
                data[:t[0], :72] = X[idx][0, :t[0],:] # person 1
                data[:t[0], 72:] = X[idx][0, :t[0],:] # person 2
                mean[:t[0], :72] = meanpose[idx][0, :t[0],:]
                mean[:t[0], 72:] = meanpose[idx][0, :t[0],:]
                rot[:t[0], :144] = rot6d[idx][0, :t[0],:] # for person 1 
                rot[:t[0], 144:] = rot6d[idx][0, :t[0],:] # for person 1
                b[:t[0], :4] = bbox[idx][0,:t[0]]
                b[:t[0], 4:] = bbox[idx][0,:t[0]]

            data_list.append(data)
            mean_list.append(mean)
            rot6d_list.append(rot)
            labels.append(Y[idx][0])
            cam_list.append(camera[idx][0])
            setup_list.append(setup_id[idx][0])
            file_list.append(file[idx][0])
            # bbox_list.append(b)
        
    print('completed: Setup id: {}, no of samples: {}'.format(jj, len(np.unique(person_idx))))
    print('==============================================================')

print('Creating Array')
data_list = np.array(data_list)
mean_list = np.array(mean_list)
labels = np.array(labels)
setup_list = np.array(setup_list)
rot6d_list = np.array(rot6d_list)
cam_list = np.array(cam_list)
file_list = np.array(file_list)
# bbox_list = np.array(bbox_list)


train_idx = np.where(setup_list%2==0)
test_idx = np.where(setup_list%2!=0)


if not os.path.isdir(main_path + '/data/'):
    os.mkdir(main_path + '/data/')

np.save(main_path + '/data/Train_File_order.npy', file_list[train_idx])
np.save(main_path + '/data/Test_File_order.npy', file_list[test_idx])

print(data_list.shape, labels.shape, setup_list.shape, rot6d_list.shape)

print('Creating Dataset')
f = h5py.File(main_path + '/data/NTU_VIBE_CSet_120.h5', 'w')
f.create_dataset('x', data=data_list[train_idx])
# f.create_dataset('test_x', data=data_list[test_idx])
print('X saved')
f.create_dataset('y', data=labels[train_idx])
# f.create_dataset('test_y', data=labels[test_idx])
print('Y saved')
f.create_dataset('mean_pose', data=mean_list[train_idx])
# f.create_dataset('test_mean_pose', data=mean_list[test_idx])
f.create_dataset('rot6d', data=rot6d_list[train_idx])
# f.create_dataset('test_rot6d', data=rot6d_list[test_idx])
print('rot6d saved')
f.create_dataset('camera', data=cam_list[train_idx])
# f.create_dataset('test_camera', data=cam_list[test_idx])
print('camera saved')
# f.create_dataset('bbox', data=bbox_list[train_idx])
# f.create_dataset('test_bbox', data=bbox_list[test_idx])
f.close()
print('dataset created..')
