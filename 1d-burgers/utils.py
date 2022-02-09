
import torch
import dgl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

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

    ax.plot(x1, y1, label=label1)
    ax.plot(x2, y2, label=label2)

    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('u(x,t)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    ax.grid()
    plt.legend(loc="upper right", facecolor='white', framealpha=1)
    fig.savefig(name, bbox_inches='tight')
    plt.show()
    


def plot_3D(U_plot, name='plot_residual.pdf'):

    font = {'size'   : 20}
    mpl.rc('font', **font)
    mpl.use('TkAgg')
    
    x = np.linspace(-1, 1, 256)
    t = np.linspace(0, 0.99, 100)
    X, T = np.meshgrid(x,t)
    
    fig, ax = plt.subplots(figsize=(16,12),subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, T, U_plot, cmap='jet',
                           linewidth=0, antialiased=True, alpha=1)


    ax.set_xlabel('x',labelpad=16)
    plt.xticks(np.arange(-1, 1.4, step=0.4))

    ax.set_ylabel('t',labelpad=16)
    ax.set_zlabel('Δu',labelpad=36)
    ax.tick_params(axis='z', which='major', pad=16)
    
    fig.colorbar(surf, ax=ax, shrink=0.75)

    plt.savefig(name, bbox_inches='tight')
    plt.show()