from MeerKATgen import Observation
from MeerKATgen.sim_params import random_SETI_params, random_RFI_params 
from MeerKATgen.create_dataset import create_simulated_obs_true_simulated
from MeerKATgen.create_dataset import create_simulated_obs_single_true_simulated
from MeerKATgen.create_dataset import create_simulated_obs_false_simulated
from MeerKATgen.create_dataset import vectorize
"""
Tests to run with github actions  
"""


print("Test Sim parameters")
RFI = random_RFI_params(10)
SETI = random_SETI_params(2)
print("Test Blank Obs")
obs = Observation(num_beams=64,
                 fchans=256,
                 tchans=16,
                 ascending=False,
                 telescope_sigma = 0.5,
                 RFI = RFI,
                 SETI = SETI,
                 obs_data=None)

data, coordinates, adj_matrix, labels  = obs.extract_all()


obs_data = data, coordinates
RFI = random_RFI_params(10)
SETI = random_SETI_params(2)
print("Test background Obs")
obs2 = Observation(num_beams=64,
                 fchans=256,
                 tchans=16,
                 ascending=False,
                 telescope_sigma = 0.5,
                 RFI = RFI,
                 SETI = SETI,
                 obs_data=obs_data)
print("batched simulation true") 
create_simulated_obs_true_simulated()
print("batched simulation false") 
create_simulated_obs_false_simulated()
print("batched simulation single true") 
create_simulated_obs_single_true_simulated()
print("passed")
print("test vectorize")
# vectorize(10, create_simulated_obs_true_simulated)
# vectorize(create_simulated_obs_false_simulated, 10)
# vectorize(create_simulated_obs_true_simulated, 10)