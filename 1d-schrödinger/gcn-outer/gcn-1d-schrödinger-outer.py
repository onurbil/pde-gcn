
import numpy as np
import torch
import torch.nn.functional as F


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import sys
import random

import dgl
from scipy import io

sys.path.append('../')
from models import GCN
from utils import create_graph, plot_2D, plot_3D

device = 'cuda' if torch.cuda.is_available() else 'cpu'


sys.path.append('../../utils')
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

nx = x.shape[0]
nt = t.shape[0]


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
    
g = create_graph(nx,nt, k=1).to(device)

edges = g.edges()
n_edges = g.number_of_edges()



degs = g.in_degrees().float()
norm = torch.pow(degs, -0.5)
norm[torch.isinf(norm)] = 0

g.ndata['norm'] = norm.unsqueeze(1).to(device)


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
    

    x_feat = inputs[:,0:1]
    t_feat = inputs[:,1:]
    x_feat.requires_grad = True 
    t_feat.requires_grad = True   
    
    
    u_pred, v_pred = net_u(x_feat,t_feat,model)
    loss_uu = loss_func(u_pred[train_bound_mask], true_u[train_bound_mask])
    loss_uv = loss_func(v_pred[train_bound_mask], true_v[train_bound_mask])


    loss_ub1 = loss_func(u_pred[bound_mask2], u_pred[bound_mask3])
    loss_vb1 = loss_func(v_pred[bound_mask2], v_pred[bound_mask3]) 

    u_pred_x = torch.autograd.grad(
        u_pred, x_feat, 
        grad_outputs=torch.ones_like(u_pred),
        retain_graph=True,
        create_graph=True,
    )[0]
         
    v_pred_x = torch.autograd.grad(
        v_pred, x_feat, 
        grad_outputs=torch.ones_like(v_pred),
        retain_graph=True,
        create_graph=True,
    )[0]

         
    loss_ub2 = loss_func(u_pred_x[bound_mask2], u_pred_x[bound_mask3])
    loss_vb2 = loss_func(v_pred_x[bound_mask2], v_pred_x[bound_mask3])     
    
    
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

MODEL_PATH = 'model-1d-schrödinger-gcn.pt'
hidden = 256
num_feat = inputs.shape[1]
out_feat = true_u.shape[1] + true_v.shape[1]



# Create mask for boundary conditions:
bound_mask = np.zeros(nx*nt)
bound_mask[:nx] = 1

train_bound_mask = torch.BoolTensor(bound_mask)
train_bound_mask[train_size:] = 0


bound_mask2 = np.zeros(nx*nt)
bound_mask2[::nx] = 1
bound_mask2 = torch.BoolTensor(bound_mask2)
bound_mask3 = np.zeros(nx*nt)
bound_mask3[nx-1::nx] = 1
bound_mask3 = torch.BoolTensor(bound_mask3)


model = GCN(g=g, in_feats=num_feat, hidden_feats=hidden, out_feats=out_feat, activation=F.tanh, dropout=0.0).to(device)

torch.set_printoptions(precision=8)
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



def plot(x1,y1, x2,y2,name='plot.pdf'):

    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 11}

    mpl.rc('font', **font)

    mpl.use('TkAgg')
    sns.set_style("dark")
    fig, ax = plt.subplots()

    ax.plot(x1, y1, label="ANN")
    ax.plot(x2, y2, label="True")

    ax.set(xlabel='x', ylabel='u(x,t)')

    ax.grid()
    plt.legend(loc="upper right", facecolor='white', framealpha=1)
    fig.savefig(name, bbox_inches='tight')
    plt.show()


def test_model(model_input,true_data):

    model = GCN(g=g, in_feats=num_feat, hidden_feats=hidden, out_feats=out_feat, activation=F.tanh, dropout=0.0).to(device)
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
    
    u_pred = u_pred.detach().numpy()
    v_pred = v_pred.detach().numpy()

    
    to_print = [14,102,200]
    time = ['011','080','157']
    for i in range(len(to_print)):
        k = to_print[i]
        l = k+1
        t = time[i]

        psi_pred = np.sqrt(u_pred[k*nx:l*nx]**2 + v_pred[k*nx:l*nx]**2)
        psi = np.sqrt(true_u[k*nx:l*nx]**2 + true_v[k*nx:l*nx]**2)
        plot_2D(x_axis,psi_pred,x_axis, psi,name='t{}_gcn_1d-schrödinger_out.pdf'.format(t),label1='GCN', label2='True')


    psi_pred = np.sqrt(u_pred**2 + v_pred**2)
    psi = np.sqrt(true_u**2 + true_v**2)

    u_plot = np.array(psi-psi_pred)
    u_plot = u_plot.reshape(nt,nx)
    
    plot_3D(u_plot, name='3d-plot_gcn_1d-schrödinger.pdf')

    return None


"""
To run from terminal enter:
python gcn-1d-schrödinger-outer.py [--test]

Example for train:
python gcn-1d-schrödinger-outer.py
Example for test:
python gcn-1d-schrödinger-outer.py --test
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

