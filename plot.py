import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import time 

def plot_solution_steady(x,T,Texact,Pe,dx):
    # Plot the steady state solution
    plt.ion() # enable interactive mode
    fig,ax=plt.subplots(nrows=2,ncols=1, dpi=200)
    ax[0].plot(x, T, lw=0.8, color='b', linestyle='solid', marker = 'o', mfc='none', ms = 4, label = '$T_{computed}$')
    ax[0].plot(x, Texact, lw=1, color='black', linestyle='dashed', label = '$T_{exact}$')
    ax[0].set_title("Peclet number = " + str("%.2f" % Pe), fontsize = 8, loc='center')

    ax[0].set_xticks(np.linspace(np.min(x), np.max(x), 11, endpoint=True )) # set dx over x-axis
    ax[0].set_yticks(np.linspace(min(np.min(T),np.min(Texact)), max(np.max(Texact),np.max(T)), 5, endpoint=True))

    ax[0].xaxis.set_major_formatter('{x:5.1f}') # set decimal point format for x-axis
    ax[0].yaxis.set_major_formatter('{x:5.2f}')

    ax[0].grid(color='grey', linestyle='solid', linewidth=0.2) # set backgroung grid
    ax[0].set_xlabel('x', fontsize=10, loc='center')
    ax[0].set_ylabel('Temperature', fontsize=10)
    ax[0].legend(loc='upper right', bbox_to_anchor=(1, 1.25), fancybox=True, shadow=False, ncol=5,fontsize = 5) 

    # 2D color mesh (heatmap)
    cs = ax[1].pcolormesh(np.append(x,x[-1]+dx),[0,0.5,1], [T,T], shading='flat', vmin=min(np.min(T),0), vmax=np.max(T))
    
    # horizontal colorbar below the heatmap
    cbar = fig.colorbar(cs, location='bottom', label='$T_{computed}$')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    cbar_ticks = np.linspace(min(np.min(T),0), np.max(T), num=5, endpoint=True)
    cbar.set_ticks(cbar_ticks)
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.tight_layout()
    
    plt.show(block = True)
