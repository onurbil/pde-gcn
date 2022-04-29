
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
from utils import create_graph, plot_3D


device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(36)


# Create data:
xy_count = 26
t_count = 11
train_size = int(xy_count*xy_count*t_count*0.9) 


x_data = np.linspace(-5,5,xy_count)
y_data = np.linspace(-5,5,xy_count)
t_data = np.linspace(0,1,t_count)

x_data = np.tile(x_data,xy_count)
y_data = np.repeat(y_data,xy_count)

x_data = np.tile(x_data,t_count)
y_data = np.tile(y_data,t_count)
t_data = np.repeat(t_data,xy_count*xy_count)

X = torch.FloatTensor(x_data[:,None])
Y = torch.FloatTensor(y_data[:,None])
T = torch.FloatTensor(t_data[:,None])

inputs = torch.cat([X,Y,T], dim=1).to(device)

def exact_sol(inp):
    
    solution = 1j*torch.exp(inp[:,2]*1j)/(torch.cosh(inp[:,0])*torch.cosh(inp[:,1]))
    re = solution.real
    im = solution.imag
    re = torch.reshape(re, (-1,1)) 
    im = torch.reshape(im, (-1,1))
    
    return re, im

true_re, true_im = exact_sol(inputs)
true_re = true_re.to(device)
true_im = true_im.to(device)

    
g = create_graph(xy_count,xy_count, t_count,k=1).to(device)

edges = g.edges()
n_edges = g.number_of_edges()


degs = g.in_degrees().float()
norm = torch.pow(degs, -0.5)
norm[torch.isinf(norm)] = 0 

g.ndata['norm'] = norm.unsqueeze(1).to(device)


def net_u(x_feat,y_feat,t_feat,model):  

    input_concat = torch.cat([x_feat,y_feat,t_feat], dim=1)
    out = model(input_concat)
    out_re = out[:,0]
    out_im = out[:,1]
    out_re = torch.reshape(out_re, (-1,1)) 
    out_im = torch.reshape(out_im, (-1,1))

    return out_re, out_im


def net_f(input_data,model):
    
    x_feat = input_data[:,0:1]
    y_feat = input_data[:,1:2]
    t_feat = input_data[:,2:]
    x_feat.requires_grad = True 
    y_feat.requires_grad = True    
    t_feat.requires_grad = True   
    
    u, v = net_u(x_feat,y_feat,t_feat,model)
    
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

    u_y = torch.autograd.grad(
        u, y_feat, 
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]
    
    u_yy = torch.autograd.grad(
        u_y, y_feat, 
        grad_outputs=torch.ones_like(u_y),
        retain_graph=True,
        create_graph=True,
    )[0]

    v_y = torch.autograd.grad(
        v, y_feat, 
        grad_outputs=torch.ones_like(v),
        retain_graph=True,
        create_graph=True,
    )[0]
    
    v_yy = torch.autograd.grad(
        v_y, y_feat, 
        grad_outputs=torch.ones_like(v_y),
        retain_graph=True,
        create_graph=True,
    )[0]
    

    w = 3 - 2*torch.square(torch.tanh(x_feat)) - 2*torch.square(torch.tanh(y_feat))
    
    fu = u_t + v_xx + v_yy + w*v
    fv = v_t - u_xx - u_yy - w*u
    
    return fu, fv


def closure():

    loss_fu, loss_fv = net_f(inputs,model)
    
    loss_fu = loss_fu[:train_size]
    loss_fu = torch.mean(torch.pow(loss_fu , 2)) 
    loss_fv = loss_fv[:train_size]
    loss_fv = torch.mean(torch.pow(loss_fv, 2))
    loss_f = loss_fu + loss_fv
    
    x_bound = inputs[:,0:1]
    y_bound = inputs[:,1:2]
    t_bound = inputs[:,2:3]
    
    u_pred, v_pred = net_u(x_bound,y_bound,t_bound,model)
    loss_uu = loss_func(u_pred[train_bound_mask], true_re[train_bound_mask])
    loss_uv = loss_func(v_pred[train_bound_mask], true_im[train_bound_mask])
    loss_u = loss_uu + loss_uv

    loss = loss_f + loss_u
    
    optimizer.zero_grad()

    loss.backward()
    if loss < best_loss:
        torch.save(model.state_dict(), MODEL_ENS_PATH)

    iter = optimizer.state_dict()['state'][0]['n_iter']
    f_eval = optimizer.state_dict()['state'][0]['func_evals']
    
    print('i:',iter,'Func.Eval:', f_eval,'Tot.Loss:', loss.data,'Func.Loss:', loss_f.data,'Bound.Loss:', loss_u.data)

    return loss


# Create a train boundry mask:
bound_mask = np.zeros((t_count,xy_count,xy_count))
bound_mask[0,:,:] = 1
bound_mask = bound_mask.reshape(-1)
train_bound_mask = torch.BoolTensor(bound_mask)
train_bound_mask[train_size:] = 0


"""
Parameters:
"""

hidden = 50
num_feat = inputs.shape[1]
out_feat = true_re.shape[1] + true_im.shape[1]


model1 = ANN_Model(num_feat, hidden, out_feat).to(device)
MODEL_PATH_ANN = "../ann-outer/model-2d-schrödinger-ann.pt"
model1.load_state_dict(torch.load(MODEL_PATH_ANN, map_location=torch.device('cpu')))


"""
Parameters:
"""

hidden_gcn = 18
num_feat = inputs.shape[1]

model2 = GCN(g=g, in_feats=num_feat, hidden_feats=hidden_gcn, out_feats=out_feat, activation=F.tanh, dropout=0.0).to(device)
MODEL_PATH_GCN = "../gcn-outer/model-2d-schrödinger-gcn.pt"
model2.load_state_dict(torch.load(MODEL_PATH_GCN, map_location=torch.device('cpu')))

        
ens_hidden = 16
    
model = Ensemble(model1, model2, hidden_units=ens_hidden).to(device)

torch.set_printoptions(precision=8)
best_loss = 99999

optimizer = torch.optim.LBFGS(
    model.parameters(), 
    lr=1.0, 
    max_iter = 100000,
    history_size = 50,
    tolerance_grad = 1e-9,
    tolerance_change = 1e-11,
    line_search_fn="strong_wolfe")

loss_func = torch.nn.MSELoss()

MODEL_ENS_PATH = 'model-2d-schrödinger-ens.pt'


def test_model(true_re,true_im):

    model.load_state_dict(torch.load(MODEL_ENS_PATH, map_location=torch.device('cpu')))
    model.eval()
    pred = model(inputs)
    u_pred = pred[:,0:1]
    v_pred = pred[:,1:2]


    test_loss_u = loss_func(u_pred[train_size:], true_re[train_size:])
    test_loss_v = loss_func(v_pred[train_size:], true_im[train_size:])

    test_loss = test_loss_u + test_loss_v      
    print('test loss:', test_loss.item())

    psi_pred = torch.sqrt(u_pred[train_size:]**2 + v_pred[train_size:]**2)
    psi = torch.sqrt(true_re[train_size:]**2 + true_im[train_size:]**2)
    max_loss = torch.max(torch.abs(psi_pred-psi))
    print('infinite norm:', max_loss.item())

    with open('test.txt', 'w') as f:
        f.write('test loss: {}\n'.format(test_loss.item()))
        f.write('infinite norm: {}'.format(max_loss.item()))


    t_size = xy_count*xy_count

    u_pred = u_pred.detach().numpy()[-t_size:]
    v_pred = v_pred.detach().numpy()[-t_size:]

    # u_pred = u_pred.detach().numpy()[5*t_size:6*t_size]
    # v_pred = v_pred.detach().numpy()[5*t_size:6*t_size]

    u_pred = np.reshape(u_pred,(xy_count,xy_count))
    v_pred = np.reshape(v_pred,(xy_count,xy_count))

    true_re = true_re[-t_size:]
    true_im = true_im[-t_size:]
    
    # true_re = true_re[5*t_size:6*t_size]
    # true_im = true_im[5*t_size:6*t_size]

    true_re = np.reshape(true_re,(xy_count,xy_count))
    true_im = np.reshape(true_im,(xy_count,xy_count))

    psi_pred_arr = np.sqrt(u_pred**2 + v_pred**2)
    psi_arr = np.sqrt(true_re**2 + true_im**2)
    u_plot = np.array(psi_arr-psi_pred_arr)

    plot_3D(u_plot,name='3d-plot_ens_2d-schrödinger_res_1s.pdf')    


"""
Train:
"""
def main(arguments):

    if not arguments.test:
        optimizer.step(closure)
        print('###Finished!###')
    else:
        test_model(true_re, true_im)



parser = argparse.ArgumentParser()
parser.add_argument('--test', default=False, action='store_true')
arguments = parser.parse_args()

# Run main():
if __name__ == "__main__":
    main(arguments)


