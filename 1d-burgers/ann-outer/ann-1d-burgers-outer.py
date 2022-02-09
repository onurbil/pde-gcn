
import numpy as np
import torch
import torch.nn as nn

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from scipy import io

sys.path.append('../')
from models import ANN_Model
from utils import plot_2D, plot_3D


"""
ANN-Model for Burgers equation trained on CPU:
"""


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(36)

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
train_size = int(nx*nt*0.9)


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
    loss_f = loss_f[:train_size]
    loss_f = torch.mean(loss_f**2)

    x_bound = inputs[:,0:1]
    t_bound = inputs[:,1:]
    
    u_pred = net_u(x_bound,t_bound,model)
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



"""
Parameters:
"""
hidden = 20
num_feat = inputs.shape[1]
out_feat = true_values.shape[1]



# Create mask for boundary conditions:
bound_mask = np.zeros(nx*nt)
bound_mask[:nx] = 1
bound_mask[::nx] = 1
bound_mask[nx-1::nx] = 1

train_bound_mask = torch.BoolTensor(bound_mask)
train_bound_mask[train_size:] = 0


model = ANN_Model(num_feat,hidden,out_feat).to(device)

torch.set_printoptions(precision=8)
MODEL_PATH = 'model-1d-burgers-ann.pt'
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

    model = ANN_Model(num_feat,hidden,out_feat).to(device)
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


    x_axis = model_input[:nx,0].cpu()
    true_data = true_data.cpu()
    u_pred = u_pred.cpu()
    

    y00=true_data[0*nx:1*nx]
    y95=true_data[95*nx:96*nx]
    y99=true_data[99*nx:100*nx]

    u00 = u_pred[0*nx:1*nx]
    u95 = u_pred[95*nx:96*nx]
    u99 = u_pred[99*nx:100*nx]


    plot_2D(x_axis, u00,x_axis, y00,name='t000_ann.pdf',label1='ANN', label2='True')
    plot_2D(x_axis, u95,x_axis, y95,name='t095_ann.pdf',label1='ANN', label2='True')
    plot_2D(x_axis, u99,x_axis, y99,name='t099_ann.pdf',label1='ANN', label2='True')
    
    u_plot = np.array(true_data-u_pred)
    u_plot = u_plot.reshape(nt,nx)
    plot_3D(u_plot, name='3d-plot_ann_1d-burgers.pdf')

    return None

    

"""
To run from terminal enter:
python ann-1d-burgers-outer.py [--test]

Example for train:
python ann-1d-burgers-outer.py
Example for test:
python ann-1d-burgers-outer.py --test
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

