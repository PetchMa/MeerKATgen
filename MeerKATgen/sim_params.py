import numpy as np
import random
import jax 
from numba import jit

def random_SETI_params(num_signals=None, seed=None):
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
        SETI_INDEX.append(random.randint(0,64))
        seti_start_index.append(random.randint(30,220))
        seti_snr.append(100*random.random()+10)
        sign = (-1)**random.randint(0,1)
        seti_drift.append(sign*random.random()*10)
        seti_width.append(30*random.random()+30)
        seti_mean.append(30*random.random()+10)

    return [SETI_INDEX, seti_start_index, seti_snr, seti_drift,  seti_width, seti_mean]


def blank_SETI_params(num_signals=None, seed=None):
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