
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


sys.path.append('../../utils')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(36)
random.seed(36)


# Create data:
xy_count = 26
t_count = 31


x_data = np.linspace(0,1,xy_count)
y_data = np.linspace(0,1,xy_count)
t_data = np.linspace(0,3,t_count)

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
    
    solution = 1/(1+torch.exp((inp[:,0]+inp[:,1]-inp[:,2])/0.2))
    solution = torch.reshape(solution, (-1,1)) 
    
    return solution

true_values = exact_sol(inputs)
true_values = true_values.to(device)

    
g = create_graph(xy_count,xy_count, t_count, k=1).to(device)

edges = g.edges()
n_edges = g.number_of_edges()



degs = g.in_degrees().float()
norm = torch.pow(degs, -0.5)
norm[torch.isinf(norm)] = 0

g.ndata['norm'] = norm.unsqueeze(1).to(device)


def net_u(x_feat,y_feat,t_feat,graph_edges,model):  

    input_concat = torch.cat([x_feat,y_feat,t_feat], dim=1)
    out = model(input_concat)

    return out


def net_f(input_data, graph_edges,model):
    
    x_feat = input_data[:,0:1]
    y_feat = input_data[:,1:2]
    t_feat = input_data[:,2:]
    x_feat.requires_grad = True 
    y_feat.requires_grad = True    
    t_feat.requires_grad = True   
    
    u = net_u(x_feat,y_feat,t_feat,graph_edges,model)
    
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

    const = 0.1
    f = u_t + u*(u_x+u_y) - const*(u_xx + u_yy)
    
    return f


def closure():

    loss_f = net_f(inputs, edges,model)
    
    loss_f = loss_f[tr_samp]
    loss_f = torch.mean(loss_f**2)
    
    x_bound = inputs[:,0:1]
    y_bound = inputs[:,1:2]
    t_bound = inputs[:,2:3]
    
    u_pred = net_u(x_bound,y_bound,t_bound,edges,model)
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


"""
ANN:
"""

hidden = 20
num_feat = inputs.shape[1]
out_feat = true_values.shape[1]


model1 = ANN_Model(num_feat, hidden, out_feat).to(device)
MODEL_PATH1 = "../ann-inner/model-2d-burgers-ann.pt"
model1.load_state_dict(torch.load(MODEL_PATH1, map_location=torch.device('cpu')))



#####

"""
Parameters:
"""

hidden_gcn = 12
num_feat = inputs.shape[1]



model2 = GCN(g=g, in_feats=num_feat, hidden_feats=hidden_gcn, out_feats=out_feat, activation=F.tanh, dropout=0.0).to(device)
MODEL_PATH2 = "../gcn-inner/model-2d-burgers-gcn.pt"
model2.load_state_dict(torch.load(MODEL_PATH2, map_location=torch.device('cpu')))



def get_train_samples(nx,ny,nt,samp_size):
    samples = random.sample(range(0, nx*ny), samp_size)
    test = np.ones(nx*ny)
    test[samples]=0
    test = np.tile(test,nt)
    test[:nx*ny]=1
    return test


tr_samp = get_train_samples(xy_count,xy_count,t_count,70)
te_samp =  1-tr_samp
tr_samp = torch.BoolTensor(tr_samp)
te_samp = torch.BoolTensor(te_samp)


# Create a train boundry mask:
bound_mask = np.zeros((t_count,xy_count,xy_count))
bound_mask[0,:,:] = 1
bound_mask = bound_mask.reshape(-1)
train_bound_mask = torch.BoolTensor(bound_mask)


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

MODEL_ENS_PATH = 'model-2d-burgers-ens.pt'


def test_model():

    model.load_state_dict(torch.load(MODEL_ENS_PATH, map_location=torch.device('cpu')))
    model.eval()
    u_pred= model(inputs)


    test_loss = loss_func(u_pred[te_samp], true_values[te_samp]) 
    max_loss = torch.max(torch.abs(u_pred[te_samp]-true_values[te_samp]))

 
    print('test loss:', test_loss.item())
    print('infinite norm:', max_loss.item())


    with open('test.txt', 'w') as f:
        f.write('test loss: {}\n'.format(test_loss.item()))
        f.write('infinite norm: {}'.format(max_loss.item()))


    t_size = xy_count*xy_count
    u_pred = u_pred.cpu().detach().numpy()

    u_pred2 = u_pred[-t_size:]
    u_pred2 = np.reshape(u_pred2,(xy_count,xy_count))

    true = true_values[-t_size:]
    true = np.reshape(true,(xy_count,xy_count))
    u_plot = true - u_pred2

    u_plot = u_plot.detach().cpu().numpy()

    plot_3D(u_plot,name='3d-plot_ens_2d-burgers.pdf')  


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




