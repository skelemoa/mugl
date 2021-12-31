import ipywidgets as widgets
import ipyvolume as ipv
from IPython.display import display
import numpy as np
import os
import h5py
import cv2

import torch 
import torch.nn as nn
from torch.autograd import Variable

from model.model import *
from model.view2hot import *
from rotation.rotation import rot6d_to_rotmat, batch_rigid_transform



# default parameters
data_idx = 'Generated'
class_idx = 'salute'
viewpoint_idx = 'Viewpoint 1'
sample_idx = 1
model_idx = 'model_199.pt'
model_names = os.listdir('./checkpoints/')
model_names.sort(key = lambda x: int(x.split(".")[0].split('_')[1]))
p1 = None
p2 = None

class_name = np.load('./files/classes.npy') # import class names
# importing 2 person classes
person_2 = ["punch_slap", "kicking", "pushing", "pat on back", "point finger", "hugging",
             "giving object", "touch pocket", "shaking hands", "walking towards", "walking apart",
             "hit with object", "wield knife", "knock over", "grab stuff", "shoot with gun", "step on foot",
             "high-five", "cheers and drink", "carry object", "take a photo", "follow", "whisper", "exchange things",
             "support somebody", "rock-paper-scissors"]

# SMPL = np.load("./files/SMPLX_NEUTRAL.npz")
# parent_array = SMPL['kintree_table'][0][:24]
# parent_array[0] = -1

parent_array = np.array([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 15, 15])
skeleton = np.load('./files/skeleton.npy')

latent_dim = 352
device = torch.device('cuda:0')
num_class = 120

# viewpoint = np.load('./files/viewpoint.npy')
# f = h5py.File('./dataset/data/Groung_truth.h5', 'r')
# gr_x = f['x'][:]
# gr_y = f['y'][:]
# gr_root = f['root'][:]
# gr_x = gr_x.reshape((gr_x.shape[0], gr_x.shape[1], 2, 24, 3))



# load model
model = Model(num_class, latent_dim).to(device)
model.load_state_dict(torch.load('./checkpoints/' + model_idx, map_location=torch.device('cpu')))
view = View2Hot().to(device)


# defining Class drop down
action_class = widgets.Dropdown(
    options=list(class_name),
    value=str(class_idx),
    description='Class:',
    disabled=False,
)

# defining viewpoint drop down
action_viewpoint = widgets.Dropdown(
    options=['Viewpoint 1', 'Viewpoint 2', 'Viewpoint 3'],
    value=str(viewpoint_idx),
    description='Viewpoint:',
    disabled=False,
)

# defining sample drop down
action_sample = widgets.Dropdown(
    options=[0,1,2,3,4,5,6,7,8,9],
    value=sample_idx,
    description='Sample:',
    disabled=True,
)


# defining model drop down
action_model = widgets.Dropdown(
    options=model_names,
    value=model_idx,
    description='Model:',
    disabled=False,
)

# defining generate button 
btn = widgets.Button(
    description='Generate Sample',
    disabled=False,
    button_style='success',
    tooltip='Click me!',
    icon='check' # (FontAwesome names without the `fa-` prefix)
)

output = widgets.Output()



# event handler functions
def btn_eventhandler(obj):
    # event handler for generate button
    global p1, p2
    output.clear_output()
    with output:
        print('Class: {}, Viewpoint: {}, Index: {}'.format(class_idx, viewpoint_idx, sample_idx))
        print(data_idx)
        c = np.where(class_name==class_idx)
        c = int(c[0])
        v_index = int(viewpoint_idx.split(' ')[1])-1
        # if data_idx == 'Generated':
        p1, p2 = infer(model,c,v_index)
        # else:
        #     p1, p2 = get_gt(c)
        
        action_sample.disabled = False
        idx = sample_idx
        # print(idx, c)
        if class_idx in person_2:
            plot(p1[idx,:,:,:], p2[idx,:,:,:])
        else:
            plot(p1[idx,:,:,:], p2[idx,:,:,:], single=True)


action_data = widgets.Dropdown(
    options=['Generated', 'Ground Truth'],
    value=data_idx,
    description='Data:',
    disabled=False,
)


def dropdown_eventhandler(change):
    # event handler for Class drop down
    global class_idx
    action_sample.disabled = True
    class_idx = change.new


def dropdown_eventhandler_vw(change):
    # event handler for viewpoint drop down
    global viewpoint_idx
    action_sample.disabled = True
    viewpoint_idx = change.new


def dropdown_eventhandler_sample(change):
    # event handler for viewpoint drop down
    global sample_idx
    output.clear_output()
    sample_idx = change.new
    idx = sample_idx
    with output:
        print('Class: {}, Viewpoint: {}, Index: {}'.format(class_idx, viewpoint_idx, sample_idx))
        if class_idx in person_2:
            plot(p1[idx,:,:,:], p2[idx,:,:,:])
        else:
            plot(p1[idx,:,:,:], p2[idx,:,:,:], single=True)

def dropdown_eventhandler_data(change):
    # event handler for data drop down
    global data_idx
    action_sample.disabled = True
    # action_no_sample.disabled = True
    data_idx = change.new



def dropdown_eventhandler_model(change):
    # event handler for viewpoint drop down
    global model_idx
    action_sample.disabled = True
    btn.disabled = True
    model_idx = change.new
    model.load_state_dict(torch.load('./checkpoints/' + model_idx, map_location=torch.device('cpu')))
    btn.disabled = False



action_data.observe(dropdown_eventhandler_data, names='value')
action_class.observe(dropdown_eventhandler, names='value')
action_viewpoint.observe(dropdown_eventhandler_vw, names='value')
action_sample.observe(dropdown_eventhandler_sample, names='value')
action_model.observe(dropdown_eventhandler_model, names='value')
input_widgets = widgets.HBox([action_class, action_viewpoint, action_sample, btn])


# plot function
def plot(skeleton_motion1, skeleton_motion2, single=False, save_gif=False):
    '''
        This function takes 2 persons as input and plots them
        skeleton_motion1: Skeleton of person 1
        skeleton_motion2: Skeleton of person 2
        single: True if single person class else false
        save_fig: Save gif file if True
    '''
    fig = ipv.figure(width=600,height=600)
    skeleton_motion1[:,:,1] *= -1
    skeleton_motion2[:,:,1] *= -1
    skeleton_motion1[:,:,2] *= -1
    skeleton_motion2[:,:,2] *= -1

    # if not single:
    #     # skeleton_motion2[:,:,0] += 0.6
    #     s = ipv.scatter(skeleton_motion1[:,:,0],skeleton_motion1[:,:,1],skeleton_motion1[:,:,2],size=2.5,color='indigo',marker='sphere')
    #     s1 = ipv.scatter(skeleton_motion2[:,:,0],skeleton_motion2[:,:,1],skeleton_motion2[:,:,2],size=2.5,color='red',marker='sphere')
    #     anim_list = [s,s1]
    # else:
    #     s = ipv.scatter(skeleton_motion1[:,:,0],skeleton_motion1[:,:,1],skeleton_motion1[:,:,2],size=2.5,color='indigo',marker='sphere')
    #     anim_list = [s]
    anim_list = []
    for i,p in enumerate(parent_array): # Run loop for each bone
        if p == -1:
            continue
        b = ipv.plot(np.array([skeleton_motion1[:,i,0],skeleton_motion1[:,p,0]]).T,np.array([skeleton_motion1[:,i,1],skeleton_motion1[:,p,1]]).T,np.array([skeleton_motion1[:,i,2],skeleton_motion1[:,p,2]]).T ,size=10, color='darkviolet')
        if not single:
            b1 = ipv.plot(np.array([skeleton_motion2[:,i,0],skeleton_motion2[:,p,0]]).T,np.array([skeleton_motion2[:,i,1],skeleton_motion2[:,p,1]]).T,np.array([skeleton_motion2[:,i,2],skeleton_motion2[:,p,2]]).T ,size=10, color='orange')
        anim_list.append(b)
        if not single:
            anim_list.append(b1)
    
    ipv.animation_control(anim_list, interval=0.01)
    ipv.style.background_color('bright')
    ipv.style.box_off()
    ipv.style.axes_off()
    ipv.show()
    
    if save_gif:
        def slide(figure, framenr, fraction):
            for a in anim_list:
                if a.sequence_index == skeleton_motion1.shape[0]:
                    a.sequence_index = 0
                a.sequence_index += 1
        save_name = "example"        
        ipv.movie(save_name + '.gif', slide, fps=5, frames=skeleton_motion1.shape[0])



def fkt(x, mean_pose):
    # forward kinematics
    rotmat = rot6d_to_rotmat(x)
    # same mean pose across timesteps
    mean_pose = torch.tensor(mean_pose.reshape((1, -1)))
    mean_pose = mean_pose.expand((x.shape[0], x.shape[1], 72))
    mean_pose = mean_pose[:,:,:].reshape((x.shape[0]*x.shape[1],-1,3))
    rotmat = rotmat.reshape((x.shape[0]*x.shape[1],-1, 3, 3))
    pred = batch_rigid_transform(rotmat.float(),mean_pose.to(device).float(),parent_array)
    x = pred.reshape((x.shape[0], x.shape[1], -1))
    return x


def interpolate(p):
    p_list = []
    for i in range(p.shape[0]):
        intr = cv2.resize(p[i],(24,128), interpolation = cv2.INTER_CUBIC)
        p_list.append(intr)
    return np.array(p_list)



def infer(model, label,v=0,num_sample=10):
    '''
        this function generates samples
        model: model object as input
        label: calss label
        v: viewpoint
    '''
    model.eval()
    # z = torch.randn(6, latent_dim).to(device).float()
    
    # y = np.repeat(np.arange(3),2)
    # y = np.arange(num_class)
    y = np.repeat(label,num_sample)
#     rot_list = []
#     for i in y:
#         idx = np.where(label==i)
#         rot_lbl = rot[idx]
#         rand = np.random.randint(rot_lbl.shape[0])
#         rot_list.append(rot_lbl[rand])
    # rot = np.repeat(viewpoint[v:v+1], num_sample, axis=0)
    # # rot = torch.tensor(rot[:,0,:]).to(device).float()
    # rot = rot[:,0,:].reshape((rot.shape[0],48,6))
    # rot = rot[:,0,:]
    v = np.repeat(v, num_sample)
    rot = np.zeros((v.shape[0], 3))
    rot[np.arange(v.shape[0]), v] = 1
    rot = torch.tensor(rot).to(device).float()
    rot = view(rot)
    # print(rot.shape)

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
        pred_3d1 = fkt(pred[:,:,:144].contiguous(), skeleton)
        pred_3d2 = fkt(pred[:,:,144:].contiguous(), skeleton)
        # pred_3d = fkt(pred, skeleton).cpu().data.numpy()
        pred_3d1 = pred_3d1.reshape((pred_3d1.shape[0], pred_3d1.shape[1], 24,-1)).cpu().data.numpy()
        pred_3d2 = pred_3d2.reshape((pred_3d2.shape[0], pred_3d2.shape[1], 24,-1)).cpu().data.numpy()
        root_pred = root_pred.cpu().data.numpy()
        pred_3d1 = pred_3d1 + root_pred[:,:,:,:3]
        pred_3d2 = pred_3d2 + root_pred[:,:,:,3:]

        pred_3d1 = interpolate(pred_3d1)
        pred_3d2 = interpolate(pred_3d2)

        # print('Generated samples:', pred_3d1.shape, pred_3d2.shape)

        return pred_3d1, pred_3d2


def get_gt(y, num_sample=10):
    idx = np.where(gr_y == y)
    x = gr_x[idx]
    root = gr_root[idx]
    # print(x.shape, root.shape)
    l = np.random.randint(0, x.shape[0], size=(num_sample,))
    x = x[l]
    root = root[l]

    x1 = x[:,:,0,:,:] + root[:,:,0:1,:]
    x2 = x[:,:,1,:,:] + root[:,:,1:2,:]
    
    x1 = interpolate(x1)
    x2 = interpolate(x2)

    return x1, x2