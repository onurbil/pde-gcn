
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
from utils import plot_3D

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(36)

# Create data:
xy_count = 26
t_count = 31
train_size = int(xy_count*xy_count*t_count*0.9) 


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



def net_u(x_feat,y_feat,t_feat,model):  

    input_concat = torch.cat([x_feat,y_feat,t_feat], dim=1)
    out = model(input_concat)

    return out


def net_f(input_data,model):
    
    x_feat = input_data[:,0:1]
    y_feat = input_data[:,1:2]
    t_feat = input_data[:,2:]
    x_feat.requires_grad = True 
    y_feat.requires_grad = True    
    t_feat.requires_grad = True   
    
    u = net_u(x_feat,y_feat,t_feat,model)
    
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

    loss_f = net_f(inputs,model)
    
    loss_f = loss_f[:train_size]
    loss_f = torch.mean(loss_f**2)
    
    x_bound = inputs[:,0:1]
    y_bound = inputs[:,1:2]
    t_bound = inputs[:,2:3]
    
    u_pred = net_u(x_bound,y_bound,t_bound,model)
    loss_u = loss_func(u_pred[train_bound_mask], true_values[train_bound_mask])

    loss = loss_f + loss_u
    
    optimizer.zero_grad()

    loss.backward()
    if loss < best_loss:
        torch.save(model.state_dict(), MODEL_PATH)

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

hidden = 20
num_feat = inputs.shape[1]
out_feat = true_values.shape[1]


model = ANN_RES_Model(num_feat, hidden, out_feat).to(device)
torch.set_printoptions(precision=8)
MODEL_PATH = "model-2d-burgers-ann-res.pt"
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
    


def test_model(model_input,true_data):

    # Test on cpu:
    model = ANN_RES_Model(num_feat, hidden, out_feat).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        
    with torch.no_grad():
        model.eval()
        u_pred = model(model_input)
    
    test_loss = loss_func(u_pred[train_size:], true_data[train_size:]) 
    max_loss = torch.max(torch.abs(u_pred[train_size:]-true_data[train_size:]))
     
    print('test loss:', test_loss.item())
    print('infinite norm:', max_loss.item())
    
    with open('test.txt', 'w') as f:
        f.write('test loss: {}\n'.format(test_loss.item()))
        f.write('infinite norm: {}'.format(max_loss.item()))


    t_size = xy_count*xy_count
    u_pred = u_pred.cpu().detach().numpy()


    u_pred2 = u_pred[6*t_size:7*t_size]

    u_pred2 = np.reshape(u_pred2,(xy_count,xy_count))
    
    true = true_values[6*t_size:7*t_size]
    true = np.reshape(true,(xy_count,xy_count))

    u_plot =  true - u_pred2
    
    u_plot = u_plot.detach().cpu().numpy()
    
    plot_3D(u_plot,name='3d-plot_ann-res_2d-burgers.pdf')    






"""
To run from terminal enter:
python ann-res-2d-burgers-outer.py [--test]

Example for train:
python ann-res-2d-burgers-outer.py
Example for test:
python ann-res-2d-burgers-outer.py --test
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

