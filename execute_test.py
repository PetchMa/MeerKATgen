from MeerKATgen import Observation

"""
Tests to run with github actions  


"""

rfi_location = [[10,10],[100,100]]
rfi_deviation = [10,10]
rfi_start_index = [40,200]
rfi_snr = [50,40]
rfi_drift = [1,-1]
rfi_width = [10,5]
rfi_mean = 10
RFI = [rfi_location, rfi_deviation, rfi_start_index, rfi_snr, rfi_drift,  rfi_width, rfi_mean]


SETI_INDEX = [0,20]
seti_start_index = [100,150]
seti_snr = [30,30]
seti_drift = [-2,2]
seti_width = [30,30]
SETI = [SETI_INDEX, seti_start_index, seti_snr, seti_drift,  seti_width]

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

rfi_location = [[0,0],[1,-3], [-5,6],[10,11]]
rfi_deviation = [1,10,2,20]
rfi_start_index = [40,200,200,100]
rfi_snr = [50,40, 30, 30]
rfi_drift = [1,-1, 1, 3]
rfi_width = [10,5, 10,20]
rfi_mean = 10
RFI = [rfi_location, rfi_deviation, rfi_start_index, rfi_snr, rfi_drift,  rfi_width, rfi_mean]


SETI_INDEX = [0,20]
seti_start_index = [100,150]
seti_snr = [30,30]
seti_drift = [-2,2]
seti_width = [30,30]
SETI = [SETI_INDEX, seti_start_index, seti_snr, seti_drift,  seti_width]


obs2 = Observation(num_beams=64,
                 fchans=256,
                 tchans=16,
                 ascending=False,
                 telescope_sigma = 0.5,
                 RFI = RFI,
                 SETI = SETI,
                 obs_data=obs_data)
