# MeerKATgen
MeerKAT radio telescope simulation package. Designed with performance in mind and utilizes 
Just in time compile (JIT) and XLA backed vectroization for batched functions. Designed for 
geometric inference models for multibeam telescopes. 

**Under construction** 
[Read docs can be found here ](https://meerkatgen.readthedocs.io/en/latest/index.html)

## Simulation Methods
This package currently has a single major function which is to create an ```observation``` object that holds all 64 
simulated events. Here is a synthetic event simulation example:

```python
from MeerKATgen import Observation
from MeerKATgen.sim_params import random_SETI_params

#number of SETI signals to simulate
SETI = random_SETI_params(3)

obs = Observation(num_beams=64,
                 fchans=512,
                 tchans=16,
                 ascending=False,
                 SETI = SETI,
                 obs_data=None)

data, coordinates, adj_matrix, labels  = obs.extract_all()
```
We can also take in real MeerKAT observations ... however data will be provided later on when observations are released...

