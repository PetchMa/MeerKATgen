import numpy as np
import astropy
import matplotlib.pyplot as plt
from random import random
from .utils import distance, gaussian
from .utils import construct_guassian_adj, construct_distance_adj
from .utils import move_point_guassian
from .utils import generate_multiple_signal_no_background
from .utils import generate_multiple_signal_real_background, calc_rfi_snr 
from .visualizations import visualize_connection
from numba import jit
from copy import deepcopy
from astropy import units as u
import setigen as stg


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
                 telescope_sigma = None,
                 RFI = None,
                 SETI = None,
                 obs_data=None,
                 **kwargs):
        
        self.num_beams = num_beams
        self.fchans = fchans
        self.tchans = tchans 
        self.df = df
        self.dt = dt 
        self.fch1 = fch1
        self.telescope_sigma = telescope_sigma
        self.RFI = RFI # list of RFI parameters
        self.SETI = SETI 
        

        if obs_data== None:
            self.data = np.zeros((self.num_beams, self.tchans, self.fchans))
            self.labels = np.zeros((self.num_beams))
            self.coordinates = self.simulate_points(self.num_beams)
            self.adj_matrix = construct_guassian_adj(self.coordinates, self.telescope_sigma )
            # inject RFI points 
            self.generate_complete_observation_blank()
        else:
            self.data, self.coordinates = obs_data
            self.labels = np.zeros((self.num_beams)) 
            self.adj_matrix = construct_guassian_adj(self.coordinates, self.telescope_sigma )
            # inject RFI points 
            self.generate_complete_observation_real()        

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
    
    def construct_RFI_snr(self, RFI_POINT, deviation, coordinates, snr):
        """
        Generate SNR for RFI signals
        
        Parameters
        ----------
         RFI_POINT : location of RFI
         deviation : decay or deviation factor 
         coordinates: COORDINATE of beams 
         snr : list of SNR orign of RFI         
        Returns
        -------
        returns the relative/experienced RFI SNR. 
        """
        RFI_POINT =np.array(RFI_POINT)
        complete_list = []
        for i in range(len(RFI_POINT)):
            complete_list.append(calc_rfi_snr(RFI_POINT[i], deviation[i], np.array(coordinates), snr[i]))
        return np.array(complete_list)


    def generate_complete_observation_blank(self):
        """
        Generate complete stack of signals
        
        Parameters
        ----------
        num : number of beams to simulate (units normalized)            
        Returns
        -------
        returns the data but filled with signals this time. 
        """
        rfi_location, rfi_deviation, rfi_start_index, rfi_snr, rfi_drift,  rfi_width, rfi_mean = self.RFI
        SETI_INDEX, seti_start_index, seti_snr, seti_drift,  seti_width = self.SETI

        relative_RFI_SNR = np.array(self.construct_RFI_snr(rfi_location, rfi_deviation, self.coordinates, rfi_snr))
        for i in range(self.num_beams):
            rfi_snr_index = relative_RFI_SNR[:,i].tolist()
            if i in SETI_INDEX:
                start_index = deepcopy(rfi_start_index) + seti_start_index
                snr =  deepcopy(rfi_snr_index) + seti_snr
                drift =  deepcopy(rfi_drift) + seti_drift
                width =  deepcopy(rfi_width) + seti_width
                

                self.data[i,:,:] = generate_multiple_signal_no_background(start_index, 
                                    snr,
                                    drift,
                                    width,
                                    mean=rfi_mean,
                                    num_freq_chans = self.fchans,
                                    num_time_chans = self.tchans,
                                    df = 2.7939677238464355*u.Hz,
                                    dt =  18.253611008*u.s,
                                    fch1 = 6095.214842353016*u.MHz) 
                self.labels[i] = 1 # update the labels as true here 
            else:
                self.data[i,:,:] = generate_multiple_signal_no_background(rfi_start_index, 
                                    rfi_snr_index,
                                    rfi_drift,
                                    rfi_width,
                                    rfi_mean,
                                    num_freq_chans = self.fchans,
                                    num_time_chans = self.tchans,
                                    df = 2.7939677238464355*u.Hz,
                                    dt =  18.253611008*u.s,
                                    fch1 = 6095.214842353016*u.MHz) 
    
    
    
    def generate_complete_observation_real(self):
        """
        Generate complete stack of signals
        
        Parameters
        ----------
        num : number of beams to simulate (units normalized)            
        Returns
        -------
        returns the data but filled with signals this time. 
        """
        rfi_location, rfi_deviation, rfi_start_index, rfi_snr, rfi_drift,  rfi_width, rfi_mean = self.RFI
        SETI_INDEX, seti_start_index, seti_snr, seti_drift,  seti_width = self.SETI
        relative_RFI_SNR = np.array(self.construct_RFI_snr(rfi_location, rfi_deviation, self.coordinates, rfi_snr))

        for i in range(self.num_beams):
            rfi_snr_index = relative_RFI_SNR[:,i].tolist()
            if i in SETI_INDEX:
                start_index = deepcopy(rfi_start_index) + seti_start_index
                # start_index.append(seti_start_index)

                snr =  deepcopy(rfi_snr_index) + seti_snr
                # snr.append(seti_snr)

                drift =  deepcopy(rfi_drift) + seti_drift
                # drift.append(seti_drift)

                width =  deepcopy(rfi_width) + seti_width
                # width.append(seti_width)

                self.data[i,:,:] = generate_multiple_signal_real_background(start_index, 
                                    snr,
                                    drift,
                                    width,
                                    background = self.data[i,:,:],
                                    num_freq_chans = self.fchans,
                                    num_time_chans = self.tchans,
                                    df = 2.7939677238464355*u.Hz,
                                    dt =  18.253611008*u.s,
                                    fch1 = 6095.214842353016*u.MHz) 
                self.labels[i] = 1 # update the labels as true here 
            else:
                self.data[i,:,:] = generate_multiple_signal_real_background(rfi_start_index, 
                                    rfi_snr_index,
                                    rfi_drift,
                                    rfi_width,
                                    background = self.data[i,:,:],
                                    num_freq_chans = self.fchans,
                                    num_time_chans = self.tchans,
                                    df = 2.7939677238464355*u.Hz,
                                    dt =  18.253611008*u.s,
                                    fch1 = 6095.214842353016*u.MHz)
    
    def extract_all(self):
        """
        Extracts all the useful data

        Parameters
        ----------
        Returns all the data
        -------
        Returns 
        """
        return self.data, self.coordinates, self.adj_matrix, self.labels 

        

          
    