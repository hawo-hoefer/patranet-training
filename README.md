# PaTraNet
XRD pattern translation network. This is an implementation of a U-Net type
neural network capable of transforming XRD patterns from $\mathrm{CuK}\alpha_{1/2}/\beta$
radiation to $\mathrm{CuK}\alpha_{1/2}$-radiation.


## Data generation
Due to licensing restraints, we cannot provide the CIFs of the analyzed phases.
We used CuO from ICSD with collection code 16025, and Fe3O4 from ICSD with 
collection code 183969. Depending on your access to ICSD, you may need to replace
them with a suitable version, perhaps from other databases like COD.

Our data generation program is available [here](https://github.com/hawo-hoefer/yaxs).
Follow the installation instructions there and install yaxs version 0.0.2.
the 
If you have a working rust toolchain on your machine, you can also just install directly
from git.
```commandline
cargo install --git https://github.com/hawo-hoefer/yaxs.git --tag 0.0.2
```


### Computing the datasets
Move your CIFs for CuO and Fe3O4 to the correct directory:
```commandline
cd /path/to/cuo-fe3o4-xrd-analysis/
mv /path/to/cuo.cif ./cif/CuO_16025.cif
mv /path/to/fe3o4.cif ./cif/Fe3O4_183969.cif
```

Simply generate the datasets using
```comandline
./generate.sh
```

## Training
First, install all the training dependencies
```commandline
python3 -m virtualenv .venv
pip install -r requirements.txt
source .venv/bin/activate
```

Then, train the network using
```commandline
python3 ./train_unet.py
```

The plots from the paper can be generated using

```commandline
python3 ./generate_plots.py
```
