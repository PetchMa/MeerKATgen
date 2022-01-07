import numpy as np

from .observation import Observation
from .sim_params import random_SETI_params
from .sim_params import blank_SETI_params 
from multiprocessing import Pool

def create_simulated_obs(num_beams,fchans, tchans, telescope_sigma , SETI, obs_data= None):
    if obs_data == None: 
        obs = Observation(num_beams=num_beams,
                    fchans=fchans,
                    tchans=tchans,
                    ascending=False,
                    telescope_sigma = telescope_sigma,
                    SETI = SETI,
                    obs_data=None)
    else:
        obs = Observation(num_beams=num_beams,
                    fchans=fchans,
                    tchans=tchans,
                    ascending=False,
                    telescope_sigma = telescope_sigma,
                    SETI = SETI,
                    obs_data=obs_data)
    return [obs.extract_all()]

def create_simulated_obs_true_simulated(num_beams=64,fchans =256, tchans =16, obs_data=None,telescope_sigma=0.5, index=0):
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


def create_simulated_obs_single_true(num_beams,fchans, tchans, telescope_sigma , SETI, obs_data=None, index=0):
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

def create_simulated_obs_false(num_beams,fchans, tchans, telescope_sigma , obs_data = None, index=0):
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

def create_simulated_obs_false_simulated(num_beams=64,fchans =256, tchans =16, obs_data=None,telescope_sigma=0.5, index=0):
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


def create_simulated_obs_false_empty(num_beams,fchans, tchans, telescope_sigma, obs_data=None, index=0):
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
    a_pool = Pool(cores)
    result = a_pool.map(func, range(num))
    return result
