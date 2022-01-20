import numpy as np
import astropy
import matplotlib.pyplot as plt
from random import random
from .utils import distance, gaussian
from .utils import construct_guassian_adj, construct_distance_adj
from .utils import move_point_guassian
from .utils import generate_multiple_signal_no_background
from .utils import generate_multiple_signal_real_background 
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
                 SETI = None,
                 obs_data=None,
                 telescope_beam_width = 0.6,
                 beamlet_width = 0.1,
                 **kwargs):
        
        self.num_beams = num_beams # number of beams 
        self.fchans = fchans    # number of frequency bins
        self.tchans = tchans  # number of time bins
        self.df = df # resolution in freq 
        self.dt = dt # resolution in time 
        self.fch1 = fch1 # start frequency 
        self.telescope_beam_width =telescope_beam_width #global beam width
        self.beamlet_width = beamlet_width  # individual beam width 
        self.SETI = SETI 
        

        if obs_data== None:
            self.data = np.zeros((self.num_beams, self.tchans, self.fchans))
            self.labels = np.zeros((self.num_beams))
            self.coordinates = self.simulate_points(self.num_beams)
            self.adj_matrix = construct_guassian_adj(self.coordinates, self.beamlet_width )
            self.generate_complete_observation_blank()
        else:
            self.data, self.coordinates = obs_data
            self.labels = np.zeros((self.num_beams)) 
            self.adj_matrix = construct_guassian_adj(self.coordinates, self.beamlet_width )
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
        SETI_INDEX, seti_start_index, seti_snr, seti_drift,  seti_width, seti_mean= self.SETI
        # we loop through each seti index and we find their coordinates 
        SETI_COORDINATE = []
        for seti_id in SETI_INDEX:
            SETI_COORDINATE.append(self.coordinates[seti_id,:])

        SETI_COORDINATE = np.array(SETI_COORDINATE)

        # we compute the distance from each seti position with the beam pointing. 
        for beam_id in range(self.num_beams):
            index =  SETI_INDEX
            start_index =  seti_start_index
            point = np.array(self.coordinates[beam_id,:])
            GLOBAL_SCALE = gaussian(distance(point, np.array([0,0])), mu=0, sig=self.telescope_beam_width)

            # we just need to expand this point to match the same dimensions with the other array. 

            distance_seti = np.zeros(len(SETI_COORDINATE))

            for i in range(len(SETI_INDEX)):
                distance_seti[i] = distance(np.array(point), SETI_COORDINATE[i])


            BEAM_SCALE =  gaussian(np.array(distance_seti), mu=0, sig=self.beamlet_width)

            snr =  list(np.array(seti_snr)*BEAM_SCALE)
            drift =  seti_drift
            width =   seti_width
            mean = seti_mean[0]

            self.data[beam_id,:,:] = generate_multiple_signal_no_background(start_index, 
                                snr,
                                drift,
                                width,
                                mean=mean,
                                num_freq_chans = self.fchans,
                                num_time_chans = self.tchans,
                                df = 2.7939677238464355*u.Hz,
                                dt =  18.253611008*u.s,
                                fch1 = 6095.214842353016*u.MHz) 

            for seti_id in SETI_INDEX:
                self.labels[seti_id] = 1 # update the labels as true here 
    # def generate_complete_observation_blank(self):
    #     """
    #     Generate complete stack of signals
        
    #     Parameters
    #     ----------
    #     num : number of beams to simulate (units normalized)            
    #     Returns
    #     -------
    #     returns the data but filled with signals this time. 
    #     """
    #     SETI_INDEX, seti_start_index, seti_snr, seti_drift,  seti_width, seti_mean= self.SETI
        
        
    #     for i in range(len(SETI_INDEX)):
    #         index =  SETI_INDEX[i]
    #         start_index =  [seti_start_index[i]]
    #         snr =  [seti_snr[i]]
    #         drift =  [seti_drift[i]]
    #         width =   [seti_width[i]]
    #         mean = seti_mean[i]

    #         self.data[index,:,:] = generate_multiple_signal_no_background(start_index, 
    #                             snr,
    #                             drift,
    #                             width,
    #                             mean=mean,
    #                             num_freq_chans = self.fchans,
    #                             num_time_chans = self.tchans,
    #                             df = 2.7939677238464355*u.Hz,
    #                             dt =  18.253611008*u.s,
    #                             fch1 = 6095.214842353016*u.MHz) 
    #         self.labels[i] = 1 # update the labels as true here 
          
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
        SETI_INDEX, seti_start_index, seti_snr, seti_drift,  seti_width, seti_mean= self.SETI
        # we loop through each seti index and we find their coordinates 
        SETI_COORDINATE = []
        for seti_id in SETI_INDEX:
            SETI_COORDINATE.append(self.coordinates[seti_id,:])

        SETI_COORDINATE = np.array(SETI_COORDINATE)

        # we compute the distance from each seti position with the beam pointing. 
        for beam_id in range(self.num_beams):
            index =  SETI_INDEX
            start_index =  seti_start_index
            point = np.array(self.coordinates[beam_id,:])
            GLOBAL_SCALE = gaussian(distance(point, np.array([0,0])), mu=0, sig=self.telescope_beam_width)

            # we just need to expand this point to match the same dimensions with the other array. 

            distance_seti = np.zeros(len(SETI_COORDINATE))

            for i in range(len(SETI_INDEX)):

                distance_seti[i] = distance(np.array(point), SETI_COORDINATE[i])
                if SETI_INDEX[i] == beam_id:
                    print(distance_seti[i])
            

            BEAM_SCALE =  gaussian(np.array(distance_seti), mu=0, sig=self.beamlet_width)

            snr =  list(np.array(seti_snr)*BEAM_SCALE)
            drift =  seti_drift
            width =   seti_width
            mean = seti_mean[0]

            self.data[beam_id,:,:] = generate_multiple_signal_real_background(start_index, 
                                snr,
                                drift,
                                width,
                                background = self.data[i,:,:],
                                num_freq_chans = self.fchans,
                                num_time_chans = self.tchans,
                                df = 2.7939677238464355*u.Hz,
                                dt =  18.253611008*u.s,
                                fch1 = 6095.214842353016*u.MHz) 

            for seti_id in SETI_INDEX:
                self.labels[seti_id] = 1
    
    # def generate_complete_observation_real(self):
    #     """
    #     Generate complete stack of signals
        
    #     Parameters
    #     ----------
    #     num : number of beams to simulate (units normalized)            
    #     Returns
    #     -------
    #     returns the data but filled with signals this time. 
    #     """
    #     SETI_INDEX, seti_start_index, seti_snr, seti_drift,  seti_width = self.SETI

    #     for i in range(self.num_beams):
     
    #         if i in SETI_INDEX:
    #             start_index = seti_start_index
    #             snr =  seti_snr
    #             drift = seti_drift
    #             width = seti_width

    #             self.data[i,:,:] = generate_multiple_signal_real_background(start_index, 
    #                                 snr,
    #                                 drift,
    #                                 width,
    #                                 background = self.data[i,:,:],
    #                                 num_freq_chans = self.fchans,
    #                                 num_time_chans = self.tchans,
    #                                 df = 2.7939677238464355*u.Hz,
    #                                 dt =  18.253611008*u.s,
    #                                 fch1 = 6095.214842353016*u.MHz) 
    #             self.labels[i] = 1 # update the labels as true here 
           
    
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

        

          
    