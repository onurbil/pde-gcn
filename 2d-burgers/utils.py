
import torch
import dgl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


def create_graph(nx,ny,nt,k=1,self_connection=True):

    v1 = []
    v2 = []
    for i in range(nx*ny*nt):

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
            
        for j in range(1,k+1):        
            if i >= j*nx*ny:
                v1.append(i)
                v2.append(i-j*nx*ny)

        for j in range(1,k+1):
            if i<(nx*ny*(nt-j)):
                v1.append(i)
                v2.append(i+j*nx*ny)            
                        
        if self_connection:
            v1.append(i)
            v2.append(i)
            
    u, v = torch.tensor(v1), torch.tensor(v2)
    graph = dgl.graph((u, v))
    return graph
    
    
def plot_3D(U_plot, name='plot_residual.pdf'):

    font = {'size'   : 20}
    mpl.rc('font', **font)
    mpl.use('TkAgg')
    
    x = np.linspace(0, 1, 26)
    y = np.linspace(0, 1, 26)

    X, Y = np.meshgrid(x,y)
    
    fig, ax = plt.subplots(figsize=(16,12),subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, U_plot, cmap='jet',
                           linewidth=0, antialiased=True, alpha=1)


    ax.set_xlabel('x',labelpad=16)
    plt.xticks(np.arange(0, 1.2, step=0.2))
    plt.yticks(np.arange(0, 1.2, step=0.2))

    ax.set_ylabel('y',labelpad=16)
    ax.set_zlabel('u(x,t) - û(x,t)',labelpad=40)
    ax.tick_params(axis='z', which='major', pad=18)
    
    fig.colorbar(surf, ax=ax, shrink=0.75, pad=0.075)

    plt.savefig(name, bbox_inches='tight')
    plt.show()