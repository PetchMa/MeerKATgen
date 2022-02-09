import numpy as np
import astropy
import matplotlib.pyplot as plt
from random import random
from numba import jit
from astropy import units as u
import setigen as stg


@jit(nopython=True)
def distance(x1,x2):
    """
    Compute Euclidean distance between two vectors

    Parameters
    ----------
    x1 : array
        Sky point for obj 1 : [ra, dec] R^2 vector 
    x2 : array
        Sky point for obj 2 : [ra, dec] R^2 vector 
    Returns
    -------
    Distance : float
        computes the euclidean norm 
    """
    return np.linalg.norm(x1-x2)

@jit(nopython=True)
def gaussian(x, mu, sig):
    """
    Compute 1-D Guassian given the value and the mean and deviation

    Parameters
    ----------
    x : float
        point value [1d]
    mu : float
        mean value 
    sig : float
        deviation 

    Returns
    -------
    guassian : float
        guassian evaluated at x 
    """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

@jit(nopython=True)
def construct_guassian_adj(coordinates, sigma_beam ):
    """
    Construct Guassian Adjacency Matrix. We do so by looping through and 
    checking the connections and computing the guassian between the points and 
    weighting the adjacency matrix that way.

    Parameters
    ----------
    coordinates : array
        n-beams of R^2 vectors 
    sigma_beam : float
        Sigma spread of beam 

    Returns
    -------
    adj_matrix : array
        Adjacency matrix weighted by guassian function
    """
    adj_matrix = np.zeros((64,64))
    for i in range(adj_matrix.shape[0]):
        x_1 = coordinates[i,:]
        for j in range(adj_matrix.shape[1]):
            x_n = coordinates[j,:]
            # We compute the distance and then use that to sample from guassian 
            # centers it at 0 to give it the guassian radius 
            adj_matrix[i,j]=gaussian(distance(x_1,x_n),0, sigma_beam)
    return adj_matrix 


@jit(nopython=True)
def construct_distance_adj(coordinates, sigma_beam, func=None ):
    """
    Construct arbitrary Adjacency Matrix 

    Parameters
    ----------
    coordinates : n-beams of R^2 vectors 
    sigma_beam : spread function of beam 
    func : python function two vectors and gives 1 scalar 
    Returns
    -------
    Distance : 
        Adjacency matrix weighted by arbitrary function 
    """
    if func != None:
        adj_matrix = np.zeros((64,64))
        for i in range(adj_matrix.shape[0]):
            x_1 = coordinates[i,:]
            for j in range(adj_matrix.shape[1]):
                x_n = coordinates[j,:]
                # We compute the distance and then use that to sample from arbitrary func 
                adj_matrix[i,j]=func(distance(x_1,x_n),0, sigma_beam)
        return adj_matrix
    else:
        adj_matrix = np.zeros((64,64))
        for i in range(adj_matrix.shape[0]):
            x_1 = coordinates[i,:]
            for j in range(adj_matrix.shape[1]):
                x_n = coordinates[j,:]
                # We compute the distance and distance is the weight
                adj_matrix[i,j]=distance(x_1,x_n)
        return adj_matrix


def move_point_guassian(coordinates, adj_matrix, point, new_location, sigma_beam):
    """
    Move point and recalculate Adjacency Matrix 

    Parameters
    ----------
    coordinates : n-beams of R^2 vectors so its [N,2] shape
    adj_matrix: adjacency matrix
    point: index of point selected
    new_location: position in normalized sky coordinates where to place point
    sigma_beam : spread function of beam 
    Returns
    -------
    Distance : 
        new set of coordinates and the new adj matrix
    """
    old_point = coordinates[point,:]
    coordinates[point,0] = new_location[0]
    coordinates[point,1] = new_location[1]
    for i in range(coordinates.shape[0]):
        adj_matrix[point, i] = gaussian(distance(coordinates[point,:],coordinates[i,:]),0, sigma_beam)
        adj_matrix[i,point] = gaussian(distance(coordinates[point,:],coordinates[i,:]),0, sigma_beam)
    return coordinates, adj_matrix


def generate_single_signal_no_background(start_index, 
                                snr,
                                drift,
                                width,
                                mean,
                                num_freq_chans = 256,
                                num_time_chans = 16,
                                df = 2.7939677238464355*u.Hz,
                                dt =  18.253611008*u.s,
                                fch1 = 900.000*u.MHz,
                                ):
    """
    generate SINGLE signal on a single beam/observation. 

    Parameters
    ----------
    start_index, 
    snr: float
        SNR of injected signal
    drift : float
        drift rate of signal
    width : float
        width of the signal in  [Hz]
    mean : float
        average background noise
    num_freq_chans : int
        number of freq channels in index
    num_time_chans : int
        number of time channels in index
    df : float
        freq resolution in the data [Hz] astropy units
    dt : float
        time resolution in data [seconds] astropy units
    fch1 : float
        start of frequency channel [mhz] astropy units
    Returns
    -------
    Distance : 
        spectrogram of fake inject data
    """   
    frame = stg.Frame(fchans=num_freq_chans*u.pixel,
                    tchans=num_time_chans*u.pixel,
                    df=df,
                    dt=dt,
                    fch1 = fch1)
    noise = frame.add_noise(x_mean=mean, noise_type='chi2')
    signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(index=start_index),
                                                drift_rate=drift*u.Hz/u.s),
                            stg.constant_t_profile(level=frame.get_intensity(snr)),
                            stg.gaussian_f_profile(width=width*u.Hz),
                            stg.constant_bp_profile(level=1))
    return frame.data

def generate_multiple_signal_no_background(start_index, 
                                snr,
                                drift,
                                width,
                                mean,
                                num_freq_chans = 256,
                                num_time_chans = 16,
                                df = 2.7939677238464355*u.Hz,
                                dt =  18.253611008*u.s,
                                fch1 = 6095.214842353016*u.MHz,
                                ):
    """
    generate MULTIPLE signal on a single beam/observation. All with synthetic backgrounds

    Parameters
    ----------
    start_index, 
    snr: float
        SNR of injected signal
    drift : float
        drift rate of signal
    width : float
        width of the signal in  [Hz]
    mean : float
        average background noise
    num_freq_chans : int
        number of freq channels in index
    num_time_chans : int
        number of time channels in index
    df : float
        freq resolution in the data [Hz] astropy units
    dt : float
        time resolution in data [seconds] astropy units
    fch1 : float
        start of frequency channel [mhz] astropy units
    Returns
    -------
    Distance : 
        spectrogram of fake inject data
    """  
    frame = stg.Frame(fchans=num_freq_chans*u.pixel,
                    tchans=num_time_chans*u.pixel,
                    df=df,
                    dt=dt,
                    fch1 = fch1)
    noise = frame.add_noise(x_mean=mean, noise_type='chi2')

    for i in range(len(snr)):
        signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(index=start_index[i]),
                                                    drift_rate=drift[i]*u.Hz/u.s),
                                stg.constant_t_profile(level=frame.get_intensity(snr[i])),
                                stg.gaussian_f_profile(width=width[i]*u.Hz),
                                stg.constant_bp_profile(level=1))
    return frame.data

def generate_single_signal_real_background(start_index, 
                                snr,
                                drift,
                                width,
                                background,
                                num_freq_chans = 256,
                                num_time_chans = 16,
                                df = 2.7939677238464355*u.Hz,
                                dt =  18.253611008*u.s,
                                fch1 = 6095.214842353016*u.MHz,):
    """
    generate SINGLE signal REAL BACKGROUND on a single beam/observation. 

    Parameters
    ----------
    start_index, 
    snr: float
        SNR of injected signal
    drift : float
        drift rate of signal
    width : float
        width of the signal in  [Hz]
    background : array
        SINGLE backgrounds to use
    mean : float
        average background noise
    num_freq_chans : int
        number of freq channels in index
    num_time_chans : int
        number of time channels in index
    df : float
        freq resolution in the data [Hz] astropy units
    dt : float
        time resolution in data [seconds] astropy units
    fch1 : float
        start of frequency channel [mhz] astropy units
    Returns
    -------
    Distance : 
        spectrogram of fake inject data
    """
    frame = stg.Frame(fchans=num_freq_chans*u.pixel,
                    tchans=num_time_chans*u.pixel,
                    df=df,
                    dt=dt,
                    fch1 = fch1,
                    data = background)
    signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(index=start_index),
                                                drift_rate=drift*u.Hz/u.s),
                            stg.constant_t_profile(level=frame.get_intensity(snr)),
                            stg.gaussian_f_profile(width=width*u.Hz),
                            stg.constant_bp_profile(level=1))
    return frame.data

def generate_multiple_signal_real_background(start_index, snr, drift, width,background,
                                                        num_freq_chans = 256,
                                                        num_time_chans = 16,
                                                        df = 2.7939677238464355*u.Hz,
                                                        dt =  18.253611008*u.s,
                                                        fch1 = 6095.214842353016*u.MHz,
                                                        ):
    """
    generate MULTIPLE signal REAL BACKGROUND on a single beam/observation. 

    Parameters
    ----------
    start_index, 
    snr: float
        SNR of injected signal
    drift : float
        drift rate of signal
    width : float
        width of the signal in  [Hz]
    background : array
        backgrounds to use
    mean : float
        average background noise
    num_freq_chans : int
        number of freq channels in index
    num_time_chans : int
        number of time channels in index
    df : float
        freq resolution in the data [Hz] astropy units
    dt : float
        time resolution in data [seconds] astropy units
    fch1 : float
        start of frequency channel [mhz] astropy units
    Returns
    -------
    Distance : 
        spectrogram of fake inject data
    """
    frame = stg.Frame(fchans=num_freq_chans*u.pixel,
                    tchans=num_time_chans*u.pixel,
                    df=df,
                    dt=dt,
                    fch1 = fch1,
                    data = background)
    for i in range(len(snr)):
        signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(index=start_index[i]),
                                                    drift_rate=drift[i]*u.Hz/u.s),
                                stg.constant_t_profile(level=frame.get_intensity(snr[i])),
                                stg.gaussian_f_profile(width=width[i]*u.Hz),
                                stg.constant_bp_profile(level=1))
    return frame.data


def calc_rfi_snr(RFI_POINT, deviation, coordinates, snr_base=30):
    """
    calculate all SNR of RFI signal

    Parameters
    ----------
    RFI_POINT : array [vector]
        single R^2 vector for where the signal originate from 
    deviation : float
        signal decay as standard deviation
    coordinates : array
        coordinates of beams 
    snr_base : float
        the base SNR of the RFI signal
    Returns
    -------
    SNR_vals : list
        list of RFI signals as a function of Guassian and distance to the point
    """   
    SNR_vals =[]
    for i in range(coordinates.shape[0]):
        SNR_vals.append(gaussian(distance(RFI_POINT, coordinates[i,:]), 0,deviation)*snr_base)
    return SNR_vals
