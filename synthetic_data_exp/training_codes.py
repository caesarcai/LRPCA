## Training codes for LRPCA 
## Implemented by Jialin Liu @ Alibaba DAMO
## Date: Dec. 07, 2021

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import scipy.io as sio
import scipy.sparse.linalg as lina
import time

## ================Preparations====================
device = torch.device('cuda:0')
datatype = torch.float32

## ================Parameters======================
r 				= 5			# underlying rank
d1 				= 1000		# size (num. of rows)
d2 				= 1000		# size (num. of columns)
alpha 			= 0.1		# fraction of outliers
step_initial 	= 0.5		# initial value of step size (eta in the paper)
ths_initial 	= 1e-3		# initial value of thresholds (zeta in the paper)
maxIt 			= 15		# num. of layers you want to train

## =============Generate RPCA problems=============
def generate_problem(r,d1,d2,alpha):
    U0_t 		= torch.randn(d1,r,dtype = datatype, device = device)/math.sqrt(d1)
    V0_t 		= torch.randn(d1,r,dtype = datatype, device = device)/math.sqrt(d2)
    idx 		= torch.randperm(d1*d2, device = device)
    idx 		= idx[:math.floor(alpha * d1*d2)]
    Y0_t 		= torch.mm(U0_t,V0_t.t())
    Y0_t 		= Y0_t.reshape(-1)
    s_range		= torch.mean(torch.abs(Y0_t))
    S0_t 		= torch.rand(len(idx), dtype = datatype, device = device)
    S0_t 		= s_range * (2.0 * S0_t - 1.0)
    Y0_t[idx] 	= Y0_t[idx] + S0_t
    Y0_t 		= Y0_t.reshape((d1,d2))
    return U0_t, V0_t, Y0_t

## ===================LRPCA model===================
class MatNet(nn.Module):
	def __init__(self):
		super(type(self),self).__init__()
		self.ths_v 		= [nn.Parameter(Variable(torch.tensor(ths_initial, dtype=datatype, device = device), requires_grad=True)) for t in range(maxIt)]
		self.step 		= [nn.Parameter(Variable(torch.tensor(step_initial, dtype=datatype, device = device), requires_grad=True)) for t in range(maxIt)]
		self.ths_backup	= [torch.tensor(0.0, dtype=datatype, device = device) for t in range(maxIt)]

	def thre(self, inputs, threshold):
		out = torch.sign(inputs) * torch.max( torch.abs(inputs) - threshold, torch.zeros([1, 1], dtype=datatype, device=device) )
		return out

	def forward(self, Y0_t, r, U0_t, V0_t, num_l):
        
		## Initialization
		S_t =  self.thre(Y0_t, self.ths_v[0])
		L, Sigma, R = torch.svd_lowrank(Y0_t-S_t, q = r, niter = 4)
		Sigsqrt = torch.diag(torch.sqrt(Sigma))
		U_t = torch.mm(L, Sigsqrt)
		V_t = torch.mm(R, Sigsqrt) 

        ## Main Loop in LRPCA
		for t in range(1, num_l):
			YmUV = Y0_t - torch.mm(U_t, V_t.t())
			S_t =  self.thre(YmUV, self.ths_v[t])
			E_t = YmUV - S_t 
			Vkernel = torch.inverse(V_t.t() @ V_t)
			Ukernel = torch.inverse(U_t.t() @ U_t)
			Unew = U_t + self.step[t] * (torch.mm(E_t,V_t) @ Vkernel)
			Vnew = V_t + self.step[t] * (torch.mm(U_t.t(),E_t).t() @ Ukernel)
			U_t = Unew
			V_t = Vnew

		## loss function in training
		loss = (torch.mm(U_t, V_t.t()) - torch.mm(U0_t, V0_t.t())).norm() 		
		return loss

	def InitializeThs(self, en_l):
		self.ths_v[en_l].data = torch.clone(self.ths_v[en_l-1].data * 0.1)
		
	def CheckNegative(self):
		isNegative = False;
		for t in range(maxIt):
			if(self.ths_v[t].data < 0):
				isNegative = True;
		if(isNegative):
			for t in range(maxIt):
				self.ths_v[t].data = torch.clone(self.ths_backup[t])
		else:
			for t in range(maxIt):
				self.ths_backup[t] = torch.clone(self.ths_v[t].data)
		return isNegative;

	def EnableSingleLayer(self,en_l):
		for t in range(maxIt): 
			self.ths_v[t].requires_grad = False
			self.step[t].requires_grad = False
		self.ths_v[en_l].requires_grad = True
		self.step[en_l].requires_grad = True

	def EnableLayers(self, num_l):
		for t in range(num_l): 
			self.ths_v[t].requires_grad = True
			self.step[t].requires_grad = True
		for t in range(num_l,maxIt): 
			self.ths_v[t].requires_grad = False
			self.step[t].requires_grad = False


## =================Training Scripts======================
Nepoches_pre 	= 500
Nepoches_full 	= 1000
lr_fac 			= 1.0															# basic learning rate

net = MatNet()
optimizers = []
for i in range(maxIt):
    optimizer = optim.SGD({net.ths_v[i]},lr = lr_fac * ths_initial / 5000.0)	# optimizer for each layer
    optimizer.add_param_group({'params': [net.step[i]], 'lr': lr_fac * 0.1})	# learning rate for each layer
    optimizers.append(optimizer)
  
## =================Layerwise Training======================
start = time.time()
for stage in range(maxIt):														# in k-th stage, we train the k-th layer
    
	## Pre-training: only train the k-th layer
	print('Layer ',stage,', Pre-training ======================')
	if(stage > 6):
		Nepoches_full = 500
	if(stage > 0):
		optimizers[stage].param_groups[0]['lr'] = net.ths_v[stage-1].data * lr_fac / 5000.0
	for epoch in range(Nepoches_pre):
		for i in range(maxIt):
			optimizers[i].zero_grad()
    
		U0_t,V0_t,Y0_t = generate_problem(r,d1,d2,alpha)
		net.EnableSingleLayer(stage)
		if(stage > 0):
			net.InitializeThs(stage)
		loss = net(Y0_t, r, U0_t, V0_t, stage+1)
		loss.backward()
		optimizers[stage].step()
    
		if(epoch % 10 == 0):
			if net.CheckNegative():
				print("Negative detected, restored")
				
		lr = optimizers[stage].param_groups[0]['lr']
		if epoch % 20 == 0:
			print("epoch: " + str(epoch), "\t loss: " + str(loss.item()))

	# Full-training: train 0~k th layers
	print('Layer ',stage,', Full-training =====================')
	if stage == 0:
		continue

	for epoch in range(Nepoches_full):
		for i in range(maxIt):
			optimizers[i].zero_grad()
    
		U0_t,V0_t,Y0_t = generate_problem(r,d1,d2,alpha)
		net.EnableLayers(stage+1)
		loss = net(Y0_t, r, U0_t, V0_t, stage+1)
		loss.backward()
        
		for i in range(stage+1):
			optimizers[i].step()

		if epoch % 20 == 0:
			print("epoch: " + str(epoch), "\t loss: " + str(loss.item()))
    
end = time.time()
print("Training end. Time: " + str(end - start))

## =====================Save model to .mat file ========================
result_ths 	= np.zeros((maxIt,))
result_stp1 = np.zeros((maxIt,))
result_stp2 = np.zeros((maxIt,))
for i in range(maxIt):
	result_ths[i] 	= net.ths_v[i].data.cpu().numpy()
	result_stp1[i] 	= net.step[i].data.cpu().numpy()

spath = 'LRPCA_alpha'+str(alpha)+'.mat'
sio.savemat(spath, {'ths':result_ths, 'step':result_stp1})










