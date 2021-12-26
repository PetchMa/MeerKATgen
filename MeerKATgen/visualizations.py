import numpy as np
import astropy
import matplotlib.pyplot as plt
from random import random


def visualize_connection(coordinates, adj_matrix):
    """
    Visualize Connections 

    Parameters
    ----------
    coordinates : Sky point for objs : [ ra, dec] R^2 vector 
    adj_matrix : connections for graph     
    Returns
    -------
    Distance : 
        Distance 
    """
  plt.figure(figsize=(8,8))
  for i in range(adj_matrix.shape[0]):
    for j in range(adj_matrix.shape[1]):
      if adj_matrix[i,j]!=0:
        x = [coordinates[i,0], coordinates[j,0]]
        y = [coordinates[i,1], coordinates[j,1]]
        plt.plot(x, y, 'ro-', linewidth=adj_matrix[i,j])
    plt.ylabel('RA [normalized]')
    plt.xlabel('DEC [normalized]')
    plt.grid()
    plt.show()
