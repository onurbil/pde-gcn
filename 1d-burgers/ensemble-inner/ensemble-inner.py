
import numpy as np
import torch

import torch.nn.functional as F
import copy

import argparse
import sys
import random

from scipy import io
import dgl

sys.path.append('../')
from models import ANN_Model, GCN, Ensemble
from utils import create_graph, plot_2D, plot_3D, plot_2D_res


device = 'cuda' if torch.cuda.is_available() else 'cpu'


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(36)
random.seed(10)

DATA_PATH = '../../data/burgers_shock.mat'
data = io.loadmat(DATA_PATH)

x = data['x'].flatten()[:,None]
t = data['t'].flatten()[:,None]
true_values = np.real(data['usol']).T

x, t = np.meshgrid(x,t)

inputs = np.hstack((x.flatten()[:,None], t.flatten()[:,None]))
true_values = true_values.flatten()[:,None]


# inputs: input data for the model. true_values: True values for output
inputs = torch.FloatTensor(inputs).to(device)
true_values = torch.FloatTensor(true_values).to(device)

# Feature space: x-t ---> 256x100
nx=256
nt=100

    
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

    return out


def net_f(input_data,model):
    
    x_feat = input_data[:,0:1]
    t_feat = input_data[:,1:]
    x_feat.requires_grad = True    
    t_feat.requires_grad = True    
        
    u = net_u(x_feat,t_feat,model)
    
    u_t = torch.autograd.grad(
        u, t_feat, 
        grad_outputs=torch.ones_like(u),
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
        
    const = 0.01/np.pi    
    f = u_t + u * u_x - const * u_xx
    
    return f


def closure():

    loss_f = net_f(inputs,model)
    loss_f = loss_f[tr_samp]
    loss_f = torch.mean(loss_f**2)

    x_bound = inputs[:,0:1]
    t_bound = inputs[:,1:]
    
    u_pred = net_u(x_bound,t_bound,model)
    loss_u = loss_func(u_pred[train_bound_mask], true_values[train_bound_mask])
    
    loss = loss_f + loss_u
    
    optimizer.zero_grad()

    loss.backward()
    if loss < best_loss:
        torch.save(model.state_dict(), MODEL_ENS_PATH)

    iter = optimizer.state_dict()['state'][0]['n_iter']
    f_eval = optimizer.state_dict()['state'][0]['func_evals']
    
    print('i:',iter,'Func.Eval:', f_eval,'Tot.Loss:', loss.data,'Func.Loss:', loss_f.data,'Bound.Loss:', loss_u.data)

            
    return loss


#####



"""
ANN:
"""
hidden = 20
num_feat = inputs.shape[1]
out_feat = true_values.shape[1]


model1 = ANN_Model(num_feat,hidden,out_feat).to(device)
MODEL_PATH1 = "../ann-inner/model-1d-burgers-ann.pt"
model1.load_state_dict(torch.load(MODEL_PATH1, map_location=torch.device('cpu')))



#####

"""
GCN:
"""


hidden_gcn = 12
num_feat = inputs.shape[1]

model2 = GCN(g=g, in_feats=num_feat, hidden_feats=hidden_gcn, out_feats=out_feat, activation=F.tanh, dropout=0.0).to(device)
MODEL_PATH2 = "../gcn-inner/model-1d-burgers-gcn.pt"
model2.load_state_dict(torch.load(MODEL_PATH2, map_location=torch.device('cpu')))



def get_train_samples(nx,nt,samp_size):
    samples = random.sample(range(0, nx), samp_size)
    test = np.ones(nx)
    test[samples]=0
    test = np.tile(test,nt)
    test[:nx]=1
    return test
    
tr_samp = get_train_samples(nx,nt,26)
te_samp =  1-tr_samp
tr_samp = torch.BoolTensor(tr_samp)
te_samp = torch.BoolTensor(te_samp)


# Create mask for boundary conditions:
bound_mask = np.zeros(nx*nt)
bound_mask[:nx] = 1
bound_mask[::nx] = 1
bound_mask[nx-1::nx] = 1

train_bound_mask = torch.BoolTensor(bound_mask)


ens_hidden = 48
    
model = Ensemble(model1, model2, hidden_units=ens_hidden).to(device)
MODEL_ENS_PATH = 'model-1d-burgers-ens.pt'
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



def test_model():

    model.load_state_dict(torch.load(MODEL_ENS_PATH, map_location=torch.device('cpu')))
    model.eval()
    u_pred = model(inputs)

    true_data = true_values.cpu()

    test_loss = loss_func(u_pred[te_samp], true_data[te_samp])
    max_loss = torch.max(torch.abs(u_pred[te_samp]-true_data[te_samp]))

    print('test loss:', test_loss.item())
    print('infinite norm:', max_loss.item())

    with open('test.txt', 'w') as f:
        f.write('test loss: {}\n'.format(test_loss.item()))
        f.write('infinite norm: {}'.format(max_loss.item()))
            

    u_pred = u_pred.cpu().detach().numpy()

    x_axis = inputs[:nx,0].cpu()

    y50=true_data[50*nx:51*nx]
    y75=true_data[75*nx:76*nx]
    y99=true_data[99*nx:100*nx]
    
    u50 = u_pred[50*nx:51*nx]
    u75 = u_pred[75*nx:76*nx]
    u99 = u_pred[99*nx:100*nx] 

    plot_2D(x_axis, u50,x_axis, y50,name='t050_ens_in.pdf',label1='GCN-FFNN', label2='True')
    plot_2D(x_axis, u75,x_axis, y75,name='t075_ens_in.pdf',label1='GCN-FFNN', label2='True')
    plot_2D(x_axis, u99,x_axis, y99,name='t099_ens_in.pdf',label1='GCN-FFNN', label2='True')

    r50 = y50-u50
    r75 = y75-u75
    r99 = y99-u99
    
    plot_2D_res(x_axis, r50,name='t050_ens_in_res.pdf',label1='Residual')
    plot_2D_res(x_axis, r75,name='t075_ens_in_res.pdf',label1='Residual')
    plot_2D_res(x_axis, r99,name='t099_ens_in_res.pdf',label1='Residual')

    u_plot = np.array(true_data-u_pred)
    u_plot = u_plot.reshape(nt,nx)
    plot_3D(u_plot, name='3d-plot_ens_1d-burgers.pdf')



"""
Train:
"""
def main(arguments):

    if not arguments.test:
        optimizer.step(closure)
        print('###Finished!###')
    else:
        test_model()


parser = argparse.ArgumentParser()
parser.add_argument('--test', default=False, action='store_true')
arguments = parser.parse_args()

# Run main():
if __name__ == "__main__":
    main(arguments)