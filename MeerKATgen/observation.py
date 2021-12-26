import numpy as np
import astropy
import matplotlib.pyplot as plt
from random import random
from utils import distance, gaussian
from utils import construct_guassian_adj, construct_distance_adj
from utils import move_point_guassian
from utils import generate_fake_background
from utils import generate_signal_real_background
from visualizations import visualize_connection
from numba import jit

class Observation(object):
    '''
    Facilitate the creation of entirely synthetic radio data 
    of 64 anntena beams 
    '''
    def __init__(self,
                 num_beams=None,
                 
                 fchans=None,
                 tchans=None,
                 df=2.7939677238464355*u.Hz,
                 dt=18.253611008*u.s,
                 fch1=6*u.GHz,
                 ascending=False,
                 data=None,
                 **kwargs):
    
    @jit
    def simulate_points(self, num):
         """
        Generate random points
        
        Parameters
        ----------
        num : number of beams to simulate (units normalized)            
        Returns
        -------
        coordinates : numpy array [num, 2]
            coordinates of where the beams will be in the sky
        """
        coordinates  = np.zeros((num, 2))
        for i in range(num):
            coordinates[i,0] = 2*random()-1
            coordinates[i,1] = 2*random()-1
        return coordinates



    def model_RFI(self, RFI_POINT, sigma, simulated_points):
        size = 1000
        sigma_x = 10
        sigma_y = sigma_x

        x = np.linspace(-1,1, size)
        y = np.linspace(-1, 1, size)
        combine = np.zeros((size,2))
        for i in range(size):
            combine[i,:] = distance([x[i],y[i]], RFI_POINT)
        x_i = combine[:,0]
        y_i = combine[:,1]
        x_i,y_i = np.meshgrid(x_i,y_i)
        z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x_i**2/(2*sigma_x**2)
            + y_i**2/(2*sigma_y**2))))
        plt.figure(figsize=(10,8))
        
        plt.contourf(x, y, z, cmap='Reds')
        plt.scatter(simulated_points[:,0],simulated_points[:,1], s=5)
        plt.grid()
        plt.show()   
    