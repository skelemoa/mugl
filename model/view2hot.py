import os 
import numpy as np
import torch 

class View2Hot(torch.nn.Module):
	def __init__(self):
		super(View2Hot,self).__init__()
		self.view_hot_6d = torch.nn.Linear(3,6,bias=False)
		self.view_hot_6d.weight = torch.nn.Parameter(torch.tensor([[ 0.74817282,  0.21238823,  0.23340215, -0.97107834,-0.62109661,-0.10907909],
			[0.99825245,-0.03015841,-0.02794450,-0.99865264,0.05206813,0.04223032],
			[0.21054807,-0.04038969,-0.19331649,-0.98113596,0.95827895,-0.18905328]]).T)

	def forward(self,view_hot):
		view_6d = self.view_hot_6d(view_hot)
		return view_6d		

# views = np.load('viewpoint.npy').reshape((3,32,2,24,6))
# y = views[:,0,0,0,:]

# model = View2Hot()

# X = torch.eye(3)
# print("Pred:",model(X).data.numpy())
# print("GR  :",y)