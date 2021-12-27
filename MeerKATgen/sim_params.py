import numpy as np
import random
import jax 
from numba import jit
def random_RFI_params(num_signals=None, seeds=None):
    if num_signals == None:
        num_signals = random.randint(1,5) 
    rfi_mean = random.random()*10+10

    rfi_location = []
    rfi_deviation = []
    rfi_start_index = []
    rfi_snr = []
    rfi_drift = []
    rfi_width = []
    
    for i in range(num_signals):
        random_x = -5*random.random()+10
        random_y = -5*random.random()+10
        rfi_location.append([random_x, random_y])
        rfi_deviation.append(5*random.random())
        rfi_start_index.append(random.randint(30,220) )
        rfi_snr.append(100*random.random()+10)
        sign = (-1)**random.randint(0,1)
        rfi_drift.append(sign*random.random()*10)
        rfi_width.append(30*random.random()+10)
    
    return [rfi_location, rfi_deviation, rfi_start_index, rfi_snr, rfi_drift,  rfi_width, rfi_mean]


def blank_RFI_params(num_signals=None, seeds=None):
    if num_signals == None:
        num_signals = 1
    rfi_mean = random.random()*10+10

    rfi_location = []
    rfi_deviation = []
    rfi_start_index = []
    rfi_snr = []
    rfi_drift = []
    rfi_width = []
    
    for i in range(num_signals):
        random_x = 1
        random_y =1
        rfi_location.append([random_x, random_y])
        rfi_deviation.append(1)
        rfi_start_index.append(1 )
        rfi_snr.append(0)
        rfi_drift.append(1)
        rfi_width.append(1)
    
    return [rfi_location, rfi_deviation, rfi_start_index, rfi_snr, rfi_drift,  rfi_width, rfi_mean]

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
    
    for i in range(num_signals):
        SETI_INDEX.append(random.randint(0,64))
        seti_start_index.append(random.randint(30,220))
        seti_snr.append(100*random.random()+10)
        sign = (-1)**random.randint(0,1)
        seti_drift.append(sign*random.random()*10)
        seti_width.append(30*random.random()+10)

    return [SETI_INDEX, seti_start_index, seti_snr, seti_drift,  seti_width]


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