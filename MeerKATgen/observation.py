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
import math

class Observation(object):
    """
    Core Object that 
    facilitate the creation of entirely synthetic radio data of 64 anntena beams

    ...

    Attributes
    ----------
    num_beams : int
        Number of beams 
    fchans : int
        Number of frequency bins
    tchans : int
        Number of time bins
    df : float
        Resolution in freq in [Hz]
    dt : float
        Resolution in time in [seconds]
    fch1 : float
        start frequency in [MHz]
    telescope_beam_width : float
        The entire FOV full width half maximum in units of distance [sigma]
    beamlet_width : float
        the individual telescope full width half maximum in units of distance [sigma] 
    data : numpy array
        Contains the data of observations [number of beams , time , frequency]
    adj_matrix: numpy array
        Contains the adjacency matrix describes the graph of the setup. [number of beams, number of beams]
    coordinates : numpy array
        Contains the coordinates in X-Y plane [number of beams, 2]
    labels : array numpy
        Contains the labels of 1 or 0 for SETI or No SETI
    SETI : dict
        Contains dictionary for all the physical parameters to be injected. Key elements of dictionary: 
        SETI['SETI_INDEX'] 
        SETI['seti_start_index'] 
        SETI['seti_snr'] 
        SETI['seti_drift'] 
        SETI['seti_width'] 
        SETI['seti_mean'] 
    
    Methods
    -------
    simulate_points(num=64)
        Simulate 64 points in the sky given the initialized FWHM values
    generate_complete_observation_blank()
        Take the initialized values and create blank observations from it with INJECTED SETI signals
    generate_complete_observation_real()
        Take the initialized values and create  observations from REAL DATA and INJECT SETI signals
    extract_all()
        Extract all the important peices of data
    """
    def __init__(self,
                 num_beams=None,
                 fchans=None,
                 tchans=None,
                 df=2.7939677238464355*u.Hz,
                 dt=18.253611008*u.s,
                 fch1=900,
                 ascending=False,
                 SETI = None,
                 coordinates = None,
                 obs_data=None,
                 telescope_beam_width = None,
                 beamlet_width =None,
                 **kwargs):
        
        self.num_beams = num_beams # number of beams 
        self.fchans = fchans    # number of frequency bins
        self.tchans = tchans  # number of time bins
        self.df = df # resolution in freq 
        self.dt = dt # resolution in time 
        self.fch1 = fch1*1e6 # start frequency in hz
        self.coordinates = coordinates
        self.SETI = SETI 
        
        if telescope_beam_width and beamlet_width:
            self.telescope_beam_width = telescope_beam_width #global beam width
            self.beamlet_width = beamlet_width  # individual beam width 
        else:
            # FWHM_TO_SIGMA = 2*math.sqrt(2*math.log(2))
            FWHM_TO_SIGMA = 1
            self.telescope_beam_width =(0.5 * ((2.998e8 / float(self.fch1)) / 13.5)*57.2958)/FWHM_TO_SIGMA
            # convert to radians
            self.beamlet_width =  (0.5 * ((2.998e8 / float(self.fch1)) / 1000)*57.2958)/FWHM_TO_SIGMA

        print("Beam Width: ",self.beamlet_width, " sigma ---- Field of View width: ", self.telescope_beam_width , ' sigma ' )

        #############################################################################
        No_coordinate_flag = False
        try:
            temp = coordinates[0,0]
        except:
            No_coordinate_flag= True

        No_data_flag = False
        try:
            temp = obs_data[0]
        except:
            No_data_flag= True

        if No_data_flag and No_coordinate_flag:
            print("Pure Synthetic - No Data and No Coordinates")
            self.data = np.zeros((self.num_beams, self.tchans, self.fchans))
            self.labels = np.zeros((self.num_beams))
            self.coordinates = self.simulate_points(self.num_beams)
            
            self.adj_matrix = construct_guassian_adj(self.coordinates, self.beamlet_width )
            self.generate_complete_observation_blank()


        elif No_data_flag:
            print("Given Coordinates But No Data")
            self.data = np.zeros((self.num_beams, self.tchans, self.fchans))
            self.labels = np.zeros((self.num_beams))

            self.adj_matrix = construct_guassian_adj(self.coordinates, self.beamlet_width )
            self.generate_complete_observation_blank()


        else:
            print("Given Real Data and real coordinates")
            self.data, self.coordinates  = obs_data
            self.labels = np.zeros((self.num_beams)) 

            self.adj_matrix = construct_guassian_adj(self.coordinates, self.beamlet_width )
            self.generate_complete_observation_real()        

    def simulate_points(self, num):
        """
        Generate random points given number of beams

        Parameters
        ----------
        num : int
            Number of beams
      
        Returns
        -------
        coordinates : numpy Array
            Returns the sky coordinates it simulated numpy array [num, 2]
        """
        coordinates  = np.zeros((num, 2))
        for i in range(num):
            r = 2*self.telescope_beam_width*random()
            theta = 2*math.pi*random()
            coordinates[i,0] = r*math.cos(theta)
            coordinates[i,1] = r*math.sin(theta)

        return coordinates
    
    def generate_complete_observation_blank(self):
        """
        Generate complete stack of signals. This simulates the observation 
        including with SYNTHETIC background given the coordinates and inital values.

        Parameters
        ----------
        None
      
        Returns
        -------
        None

        Manipulations
        -------------
        data : Array
            Adds the simulated signal into the array
        labels : array
            Records the labels with the injected signals
        
        """
        SETI_INDEX = self.SETI['SETI_INDEX'] 
        seti_start_index = self.SETI['seti_start_index'] 
        seti_snr = self.SETI['seti_snr'] 
        seti_drift = self.SETI['seti_drift'] 
        seti_width = self.SETI['seti_width'] 
        seti_mean = self.SETI['seti_mean'] 

        # SETI_INDEX, seti_start_index, seti_snr, seti_drift,  seti_width, seti_mean= self.SETI
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

     
    def generate_complete_observation_real(self):
        """
        Generate complete stack of signals. This simulates the observation 
        using REAL background given the coordinates and inital values.

        Parameters
        ----------
        None
      
        Returns
        -------
        None

        Manipulations
        -------------
        data : Array
            Adds the simulated signal into the array
        labels : array
            Records the labels with the injected signals
        
        """
        SETI_INDEX = self.SETI['SETI_INDEX'] 
        seti_start_index = self.SETI['seti_start_index'] 
        seti_snr = self.SETI['seti_snr'] 
        seti_drift = self.SETI['seti_drift'] 
        seti_width = self.SETI['seti_width'] 
        seti_mean = self.SETI['seti_mean'] 
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

    def extract_all(self):
        """
        Extracts the data, coordinates, adjacency matrix and the labels.

        Parameters
        ----------
        None
      
        Returns
        -------
        data : array
            the observations
        coordinates : array
            the coordinates in the sky for these beams
        adj_matrix : array
            adjacency matrix for the graph
        labels : array
            labels for where the SETI signal is or isn't.
        """
        return self.data, self.coordinates, self.adj_matrix, self.labels 

        

          
    