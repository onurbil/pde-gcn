
import torch
import dgl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import io

def create_graph(nx,ny,k=1,self_connection=True):

    v1 = []
    v2 = []
    for i in range(nx*ny):

        for j in reversed(range(1,k+1)):
            if (i%(nx*ny))>=j*nx:
                v1.append(i)
                v2.append(i-j*nx)
        
        for j in reversed(range(1,k+1)):
            if i%nx >= j:
                v1.append(i)
                v2.append(i-j)  
                
        for j in range(1,k+1):
            if i%nx < (nx-j):
                v1.append(i)
                v2.append(i+j)
            
        for j in range(1,k+1):
            if (i%(nx*ny))<(nx*ny-j*nx):
                v1.append(i)
                v2.append(i+j*nx)
                                    
        if self_connection:
            v1.append(i)
            v2.append(i)
            
    u, v = torch.tensor(v1), torch.tensor(v2)
    graph = dgl.graph((u, v))
    return graph
    
    
    


def plot_2D(x1,y1, x2,y2,name='plot.pdf',label1='Pred', label2='True'):

    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 16}

    mpl.rc('font', **font)

    mpl.use('TkAgg')
    sns.set_style("dark")
    fig, ax = plt.subplots()

    ax.plot(x2, y2, label=label2)
    ax.plot(x1, y1, label=label1, linestyle='dashed')

    ax.set_xlabel('t', fontsize=16)
    ax.set_ylabel('u(x,t)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    ax.grid()
    plt.legend(facecolor='white', framealpha=1)
    fig.savefig(name, bbox_inches='tight')
    plt.show()
    

def plot_2D_res(x1,y1,name='res_plot.pdf',label1='Residual'):

    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 16}

    mpl.rc('font', **font)

    mpl.use('TkAgg')
    sns.set_style("dark")
    fig, ax = plt.subplots()

    ax.plot(x1, y1,'g', label=label1)

    ax.set_xlabel('t', fontsize=16)
    ax.set_ylabel('u(x,t) - û(x,t)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    ax.grid()
    plt.legend(facecolor='white', framealpha=1)
    fig.savefig(name, bbox_inches='tight')
    plt.show()

    
def plot_3D(U_plot, name='plot_residual.pdf'):

    font = {'size'   : 20}
    mpl.rc('font', **font)
    mpl.use('TkAgg')
    
    DATA_PATH = '../../data/NLS.mat'
    data = io.loadmat(DATA_PATH)

    x = data['x'].flatten()[:,None]
    t = data['tt'].flatten()[:,None]
    true_values = data['uu']
    # Add (5,t) to the dataset:
    x = np.append(x,[[5]],axis=0)
    X, T = np.meshgrid(x,t)
    
    fig, ax = plt.subplots(figsize=(16,12),subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, T, U_plot, cmap='jet',
                           linewidth=0, antialiased=True, alpha=1)


    ax.set_xlabel('x',labelpad=16)
    plt.xticks(np.arange(-5, 7, step=2))
    plt.yticks(np.arange(0, 1.8, step=0.3))


    ax.set_ylabel('t',labelpad=16)
    ax.set_zlabel('u(x,t) - û(x,t)',labelpad=50)
    ax.tick_params(axis='z', which='major', pad=26)
    
    fig.colorbar(surf, ax=ax, shrink=0.75)

    plt.savefig(name, bbox_inches='tight')
    plt.show()