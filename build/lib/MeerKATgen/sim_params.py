import numpy as np
import random
import jax 
from numba import jit

def random_SETI_params(num_signals=None, seed=None):
    """
    Generate SETI parameters at random 

    Parameters
    ----------
    num_signals : int
        number of signals to inject
    seed : float
        make the random deterministic
    
    Returns
    -------
    SETI : dict
        Dictionary containing all the data for parameters
    """
    if num_signals == None:
        num_signals = random.randint(1,5)  
    num_signal = num_signals 
    rfi_mean = random.random()*10+10

    SETI_INDEX = []
    seti_start_index = []
    seti_snr = []
    seti_drift = []
    seti_width = []
    seti_mean = []
    for i in range(num_signals):
        SETI_INDEX.append(random.randint(0,63))
        seti_start_index.append(random.randint(30,220))
        seti_snr.append(100*random.random()+10)
        sign = (-1)**random.randint(0,1)
        seti_drift.append(sign*random.random()*4)
        seti_width.append(30*random.random()+50)
        seti_mean.append(30*random.random()+10)

    SETI = {}
    SETI['SETI_INDEX'] = SETI_INDEX
    SETI['seti_start_index'] = seti_start_index
    SETI['seti_snr'] = seti_snr
    SETI['seti_drift'] = seti_drift
    SETI['seti_width'] = seti_width
    SETI['seti_mean'] = seti_mean
    return SETI


def blank_SETI_params(num_signals=None, seed=None):
    """
    Generate NO SETI parameters 

    Parameters
    ----------
    num_signals : int
        number of signals to inject
    seed : float
        make the random deterministic
    
    Returns
    -------
    SETI : dict
        Dictionary containing all the data for parameters [this effectively does nothing at the moment]
    """
    if num_signals == None:
        num_signals = 1
    num_signal = num_signals 
    rfi_mean = random.random()*10+10

    SETI_INDEX = []
    seti_start_index = []
    seti_snr = []
    seti_drift = []
    seti_width = []
    
    for i in range(num_signals):
        SETI_INDEX.append(0)
        seti_start_index.append(0)
        seti_snr.append(0)
        sign =0
        seti_drift.append(1)
        seti_width.append(1)

    return [SETI_INDEX, seti_start_index, seti_snr, seti_drift,  seti_width]