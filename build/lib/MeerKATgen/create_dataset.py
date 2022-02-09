import numpy as np

from .observation import Observation
from .sim_params import random_SETI_params
from .sim_params import blank_SETI_params 
from multiprocessing import Pool

def create_simulated_obs(num_beams,fchans, tchans, telescope_beam_width ,beamlet_width, SETI, obs_data= None):
    """
    Create simulated observations from given parameters

    Parameters
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
    telescope_beam_width : float
        The entire FOV full width half maximum in units of distance [sigma]
    beamlet_width : float
        the individual telescope full width half maximum in units of distance [sigma] 
    coordinates : numpy array
        Contains the coordinates in X-Y plane [number of beams, 2]
    SETI : dict
        Contains dictionary for all the physical parameters to be injected. Key elements of dictionary: 
        SETI['SETI_INDEX'] 
        SETI['seti_start_index'] 
        SETI['seti_snr'] 
        SETI['seti_drift'] 
        SETI['seti_width'] 
        SETI['seti_mean'] 
    obs_data : array
        actual sky observation
    
    Returns
    -------
    dataset : list
        list of observation metrics data, adj matrix, coordinates etc
    """
    if obs_data == None: 
        obs = Observation(num_beams=num_beams,
                    fchans=fchans,
                    tchans=tchans,
                    ascending=False,
                    telescope_beam_width = telescope_beam_width,
                    beamlet_width = beamlet_width,
                    SETI = SETI,
                    obs_data=None)
    else:
        obs = Observation(num_beams=num_beams,
                    fchans=fchans,
                    tchans=tchans,
                    ascending=False,
                    telescope_beam_width = telescope_beam_width,
                    beamlet_width = beamlet_width,
                    SETI = SETI,
                    obs_data=obs_data)
    return [obs.extract_all()]

def create_simulated_obs_true_simulated(num_beams=64,fchans =256, tchans =16, obs_data=None,telescope_sigma=0.5, index=0):
    """
    Create TRUE simulated observations from given parameters that contains SETI signals

    Parameters
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
    telescope_beam_width : float
        The entire FOV full width half maximum in units of distance [sigma]
    beamlet_width : float
        the individual telescope full width half maximum in units of distance [sigma] 
    coordinates : numpy array
        Contains the coordinates in X-Y plane [number of beams, 2]
    SETI : dict
        Contains dictionary for all the physical parameters to be injected. Key elements of dictionary: 
        SETI['SETI_INDEX'] 
        SETI['seti_start_index'] 
        SETI['seti_snr'] 
        SETI['seti_drift'] 
        SETI['seti_width'] 
        SETI['seti_mean'] 
    obs_data : array
        actual sky observation
    
    Returns
    -------
    dataset : list
        list of observation objects
    """
    SETI = random_SETI_params()   
    if obs_data:
        obs = Observation(num_beams=num_beams,
                    fchans=fchans,
                    tchans=tchans,
                    ascending=False,
                    telescope_sigma = telescope_sigma,
                    SETI = SETI,
                    obs_data=obs_data)
    else:
        obs = Observation(num_beams=num_beams,
                    fchans=fchans,
                    tchans=tchans,
                    ascending=False,
                    telescope_sigma = telescope_sigma,
                    SETI = SETI,
                    obs_data=None)
    return [obs.extract_all()]


# def create_simulated_obs_single_true(num_beams,fchans, tchans, telescope_sigma , SETI, obs_data=None, index=0):
#     if obs_data:
#         obs = Observation(num_beams=num_beams,
#                     fchans=fchans,
#                     tchans=tchans,
#                     ascending=False,
#                     telescope_sigma = telescope_sigma,
#                     SETI = SETI,
#                     obs_data=obs_data)
#     else:
#         obs = Observation(num_beams=num_beams,
#                     fchans=fchans,
#                     tchans=tchans,
#                     ascending=False,
#                     telescope_sigma = telescope_sigma,
#                     SETI = SETI,
#                     obs_data=None)
#     return [obs.extract_all()]


def create_simulated_obs_single_true_simulated(num_beams=64,fchans =256, tchans =16, obs_data=None, telescope_sigma=0.5, index=0):
    SETI = random_SETI_params()   
    if obs_data:
        obs = Observation(num_beams=num_beams,
                    fchans=fchans,
                    tchans=tchans,
                    ascending=False,
                    telescope_sigma = telescope_sigma,
                    SETI = SETI,
                    obs_data=obs_data)
    else:
        obs = Observation(num_beams=num_beams,
                    fchans=fchans,
                    tchans=tchans,
                    ascending=False,
                    telescope_sigma = telescope_sigma,
                    SETI = SETI,
                    obs_data=None)
    return [obs.extract_all()]

# def create_simulated_obs_false(num_beams,fchans, tchans, telescope_sigma , obs_data = None, index=0):
#     """
#     Create FALSE simulated observations from given parameters

#     Parameters
#     ----------
#     num_beams : int
#         Number of beams 
#     fchans : int
#         Number of frequency bins
#     tchans : int
#         Number of time bins
#     df : float
#         Resolution in freq in [Hz]
#     dt : float
#         Resolution in time in [seconds]
#     telescope_beam_width : float
#         The entire FOV full width half maximum in units of distance [sigma]
#     beamlet_width : float
#         the individual telescope full width half maximum in units of distance [sigma] 
#     obs_data : array
#         actual sky observation
    
#     Returns
#     -------
#     dataset : list
#         list of observation metrics data, adj matrix, coordinates etc
#     """
#     SETI = blank_SETI_params()
#     if obs_data:
#         obs = Observation(num_beams=num_beams,
#                     fchans=fchans,
#                     tchans=tchans,
#                     ascending=False,
#                     telescope_sigma = telescope_sigma,
#                     SETI = SETI,
#                     obs_data=obs_data)
#     else:
#         obs = Observation(num_beams=num_beams,
#                     fchans=fchans,
#                     tchans=tchans,
#                     ascending=False,
#                     telescope_sigma = telescope_sigma,
#                     SETI = SETI,
#                     obs_data=None)
#     return [obs.extract_all()]

# def create_simulated_obs_false_simulated(num_beams=64,fchans =256, tchans =16, obs_data=None,telescope_sigma=0.5, index=0):
#     """
#     Create simulated observations from given parameters

#     Parameters
#     ----------
#     num_beams : int
#         Number of beams 
#     fchans : int
#         Number of frequency bins
#     tchans : int
#         Number of time bins
#     df : float
#         Resolution in freq in [Hz]
#     dt : float
#         Resolution in time in [seconds]
#     telescope_beam_width : float
#         The entire FOV full width half maximum in units of distance [sigma]
#     beamlet_width : float
#         the individual telescope full width half maximum in units of distance [sigma] 
#     obs_data : array
#         actual sky observation
    
#     Returns
#     -------
#     dataset : list
#         list of observation metrics data, adj matrix, coordinates etc
#     """
#     SETI = blank_SETI_params()
#     if obs_data:
#         obs = Observation(num_beams=num_beams,
#                     fchans=fchans,
#                     tchans=tchans,
#                     ascending=False,
#                     telescope_sigma = telescope_sigma,
#                     SETI = SETI,
#                     obs_data=obs_data)
#     else:
#         obs = Observation(num_beams=num_beams,
#                     fchans=fchans,
#                     tchans=tchans,
#                     ascending=False,
#                     telescope_sigma = telescope_sigma,
#                     SETI = SETI,
#                     obs_data=None)
#     return [obs.extract_all()]


def create_simulated_obs_false_empty(num_beams,fchans, tchans, telescope_sigma, obs_data=None, index=0):
    """
    Create BLANK observations

    Parameters
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
    telescope_beam_width : float
        The entire FOV full width half maximum in units of distance [sigma]
    beamlet_width : float
        the individual telescope full width half maximum in units of distance [sigma] 
    obs_data : array
        actual sky observation
    
    Returns
    -------
    dataset : list
        list of observation metrics data, adj matrix, coordinates etc
    """
    SETI = blank_SETI_params()
    if obs_data:
        obs = Observation(num_beams=num_beams,
                    fchans=fchans,
                    tchans=tchans,
                    ascending=False,
                    telescope_sigma = telescope_sigma,
                    SETI = SETI,
                    obs_data=obs_data)
    else:
        obs = Observation(num_beams=num_beams,
                    fchans=fchans,
                    tchans=tchans,
                    ascending=False,
                    telescope_sigma = telescope_sigma,
                    SETI = SETI,
                    obs_data=None)
    return [obs.extract_all()]


def parallel(num, func, cores = 20):
    """
    Automatically parallelize looped operations

    Parameters
    ----------
    num : int
        Number things to repeat 
    func : python object
        function to be repeated
    cores : int
        number of cpu cores to use
    
    Returns
    -------
    dataset : list
        list of things that were executed combined together
    """
    a_pool = Pool(cores)
    result = a_pool.map(func, range(num))
    return result
