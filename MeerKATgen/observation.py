import numpy as np
import astropy
import matplotlib.pyplot as plt
from random import random
from utils import distance, gaussian
from utils import construct_guassian_adj, construct_distance_adj
from utils import move_point_guassian
from utils import generate_multiple_signal_no_background
from utils import generate_multiple_signal_real_background, calc_rfi_snr 
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
        
        self.num_beams = num_beams
        self.fchans = fchans
        self.tchans = tchans 
        self.df = df
        self.dt = dt 
        self.fch1 = fch1
        self.telescope_sigma = telescope_sigma
        self.RFI_POINTS = RFI_POINTS # list of RFI parameters

        self.data = np.zeros((self.num_beams, self.tchans, self.fchans))
        self.coordinates = self.simulate_points(self.num_beams)
        self.adj_matrix = construct_guassian_adj(self.coordinates, self.telescope_sigma )
        # inject RFI points 
        if self.RFI_POINTS != None:
            for POINT in self.RFI_POINTS:
                #point holds data about the RFI parameters
                rfi_location, rfi_deviation, start_index, snr, drift,  width, mean = POINT
                


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

    def create_RFI_observations(self, RFI_POINT, deviation, coordinates, snr_base=30):
        
        

          
    