
import numpy as np
import torch

from scipy import io

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import sys
import random

sys.path.append('../')
from models import ANN_RES_Model
from utils import plot_2D, plot_3D

device = 'cuda' if torch.cuda.is_available() else 'cpu'



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(36)


DATA_PATH = '../../data/NLS.mat'
data = io.loadmat(DATA_PATH)


x = data['x'].flatten()[:,None]
t = data['tt'].flatten()[:,None]
true_values = data['uu']
# Add (5,t) to the dataset:
x = np.append(x,[[5]],axis=0)
true_values = np.append(true_values, true_values[:1,:], axis=0)
#
nx=x.shape[0]
nt=t.shape[0]


bound_x1=torch.FloatTensor(np.array(5.0).repeat(nt).reshape(nt,1))
bound_x2=torch.FloatTensor(np.array(-5.0).repeat(nt).reshape(nt,1))
bound_data1 = torch.cat([bound_x1,torch.FloatTensor(t)], dim=1).to(device)
bound_data2 = torch.cat([bound_x2,torch.FloatTensor(t)], dim=1).to(device)


true_u = np.real(true_values).T
true_v = np.imag(true_values).T


x, t = np.meshgrid(x,t)

inputs = np.hstack((x.flatten()[:,None], t.flatten()[:,None]))
true_u = true_u.flatten()[:,None]
true_v = true_v.flatten()[:,None]

inputs = torch.FloatTensor(inputs).to(device)
true_u = torch.FloatTensor(true_u).to(device)
true_v = torch.FloatTensor(true_v).to(device)


train_size = int(nx*nt*0.9)


def net_u(x_feat,t_feat,model):  

    input_concat = torch.cat([x_feat,t_feat], dim=1)
    out = model(input_concat)
    out_re = out[:,0]
    out_im = out[:,1]
    out_re = torch.reshape(out_re, (-1,1)) 
    out_im = torch.reshape(out_im, (-1,1))

    return out_re, out_im


def net_f(input_data,model):
    
    x_feat = input_data[:,0:1]
    t_feat = input_data[:,1:]
    x_feat.requires_grad = True 
    t_feat.requires_grad = True   
    
    u, v = net_u(x_feat,t_feat,model)
    
    u_t = torch.autograd.grad(
        u, t_feat, 
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]
    
    v_t = torch.autograd.grad(
        v, t_feat, 
        grad_outputs=torch.ones_like(v),
        retain_graph=True,
        create_graph=True,
    )[0]
    
    u_x = torch.autograd.grad(
        u, x_feat, 
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]

    u_xx = torch.autograd.grad(
        u_x, x_feat, 
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True,
    )[0]    

    v_x = torch.autograd.grad(
        v, x_feat, 
        grad_outputs=torch.ones_like(v),
        retain_graph=True,
        create_graph=True,
    )[0]

    v_xx = torch.autograd.grad(
        v_x, x_feat, 
        grad_outputs=torch.ones_like(v_x),
        retain_graph=True,
        create_graph=True,
    )[0]   

    
    fu = u_t + 0.5*v_xx + (u**2+v**2)*v
    fv = v_t - 0.5*u_xx - (u**2+v**2)*u
    
    return fu, fv


def closure():

    loss_fu, loss_fv = net_f(inputs,model)
    
    loss_fu = loss_fu[:train_size]
    loss_fu = torch.mean(torch.pow(loss_fu , 2)) 
    loss_fv = loss_fv[:train_size]
    loss_fv = torch.mean(torch.pow(loss_fv, 2))
    loss_f = loss_fu + loss_fv
    
    x_bound1 = inputs[:,0:1]
    t_bound1 = inputs[:,1:2]
    
    x_bound2 = bound_data1[:,0:1]
    t_bound2 = bound_data1[:,1:2]
    x_bound2.requires_grad = True 
    t_bound2.requires_grad = True       
    x_bound3 = bound_data2[:,0:1]
    t_bound3 = bound_data2[:,1:2]
    x_bound3.requires_grad = True 
    t_bound3.requires_grad = True     
    
    
    
    
    u_pred, v_pred = net_u(x_bound1,t_bound1,model)
    loss_uu = loss_func(u_pred[train_bound_mask], true_u[train_bound_mask])
    loss_uv = loss_func(v_pred[train_bound_mask], true_v[train_bound_mask])

    
    u_b1, v_b1 = net_u(x_bound2,t_bound2,model)
    u_b2, v_b2 = net_u(x_bound3,t_bound3,model)
    loss_ub1 = loss_func(u_b1, u_b2)
    loss_vb1 = loss_func(v_b1, v_b2) 

    

    u_x_b1 = torch.autograd.grad(
        u_b1, x_bound2, 
        grad_outputs=torch.ones_like(u_b1),
        retain_graph=True,
        create_graph=True,
    )[0]       
    v_x_b1 = torch.autograd.grad(
        v_b1, x_bound2, 
        grad_outputs=torch.ones_like(v_b1),
        retain_graph=True,
        create_graph=True,
    )[0]      
    u_x_b2 = torch.autograd.grad(
        u_b2, x_bound3, 
        grad_outputs=torch.ones_like(u_b2),
        retain_graph=True,
        create_graph=True,
    )[0]       
    v_x_b2 = torch.autograd.grad(
        v_b2, x_bound3, 
        grad_outputs=torch.ones_like(v_b2),
        retain_graph=True,
        create_graph=True,
    )[0]  
    
    
         
    loss_ub2 = loss_func(u_x_b1, u_x_b2)
    loss_vb2 = loss_func(v_x_b1, v_x_b2)     
    
    
    loss_u = loss_uu + loss_uv + loss_ub1 + loss_vb1 + loss_ub2 + loss_vb2
    loss = loss_f + loss_u
    
    optimizer.zero_grad()

    loss.backward()
    if loss < best_loss:
        torch.save(model.state_dict(), MODEL_PATH)

    iter = optimizer.state_dict()['state'][0]['n_iter']
    f_eval = optimizer.state_dict()['state'][0]['func_evals']
    
    print('i:',iter,'Func.Eval:', f_eval,'Tot.Loss:', loss.data,'Func.Loss:', loss_f.data,'Bound.Loss:', loss_u.data)
    
    return loss



"""
Parameters:
"""
hidden = 100
num_feat = inputs.shape[1]
out_feat = true_u.shape[1] + true_v.shape[1]


# Create mask for boundary conditions:
bound_mask = np.zeros(nx*nt)
bound_mask[:nx] = 1

train_bound_mask = torch.BoolTensor(bound_mask)
train_bound_mask[train_size:] = 0


model = ANN_RES_Model(num_feat, hidden, out_feat).to(device)

torch.set_printoptions(precision=8)
MODEL_PATH = 'model-1d-schrödinger-ann-res.pt'
best_loss = 99999

optimizer = torch.optim.LBFGS(
    model.parameters(), 
    lr = 1.0, 
    max_iter = 50000, 
    history_size = 50,
    tolerance_grad = 1e-9,
    tolerance_change = 1e-11,
    line_search_fn = 'strong_wolfe')

loss_func = torch.nn.MSELoss()



def test_model(model_input,true_data):

    model = ANN_RES_Model(num_feat, hidden, out_feat).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))


    with torch.no_grad():
        model.eval()
        pred = model(model_input)
    

    u_pred = pred[:,:1]
    v_pred = pred[:,1:2]

    test_loss_u = loss_func(u_pred[train_size:], true_u[train_size:])
    test_loss_v = loss_func(v_pred[train_size:], true_v[train_size:])
    test_loss = test_loss_u + test_loss_v      
    print('test loss:', test_loss.item())

    psi_pred = np.sqrt(u_pred[train_size:]**2 + v_pred[train_size:]**2)
    psi = np.sqrt(true_u[train_size:]**2 + true_v[train_size:]**2)
    max_loss = torch.max(torch.abs(psi_pred-psi))
    print('infinite norm:', max_loss.item())

    with open('test.txt', 'w') as f:
        f.write('test loss: {}\n'.format(test_loss.item()))
        f.write('infinite norm: {}'.format(max_loss.item()))

    
    x_axis = model_input[:nx,0]

    to_print = [14,102,200]
    time = ['011','080','157']
    for i in range(len(to_print)):
        k = to_print[i]
        l = k+1
        t = time[i]

        psi_pred = np.sqrt(u_pred[k*nx:l*nx]**2 + v_pred[k*nx:l*nx]**2)
        psi = np.sqrt(true_u[k*nx:l*nx]**2 + true_v[k*nx:l*nx]**2)
        plot_2D(x_axis,psi_pred,x_axis, psi,name='t{}_ann-res_1d-schrödinger_out.pdf'.format(t),label1='ANN-RES', label2='True')

    psi_pred = np.sqrt(u_pred**2 + v_pred**2)
    psi = np.sqrt(true_u**2 + true_v**2)

    u_plot = np.array(psi-psi_pred)
    u_plot = u_plot.reshape(nt,nx)
    
    plot_3D(u_plot, name='3d-plot_ann-res_1d-schrödinger.pdf')

    return None



"""
To run from terminal enter:
python ann-res-1d-schrödinger-outer.py [--test]

Example for train:
python ann-res-1d-schrödinger-outer.py
Example for test:
python ann-res-1d-schrödinger-outer.py --test
"""


"""
Train:
"""
def main(arguments):

    if not arguments.test:
        optimizer.step(closure)
        print('###Finished!###')
    else:
        test_model(inputs,true_values)

parser = argparse.ArgumentParser()
parser.add_argument('--test', default=False, action='store_true')
arguments = parser.parse_args()

# Run main():
if __name__ == "__main__":
    main(arguments)

