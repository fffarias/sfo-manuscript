# Applying a stochastic quasi-Newton optimizer to least-squares reverse time migration

 
This repository demonstrates how to use the [Sum of Functions Optimizer (SFO)](https://github.com/Sohl-Dickstein/Sum-of-Functions-Optimizer) applied to seismic imaging, more specifically to the least-squares reverse time migration (LSRTM) to reproduce the stochastic inversion in the Marmousi model presented in the manuscript entitled _Applying a stochastic quasi-Newton optimizer to least-squares reverse time migration_, sent to Computers and Geosciences. Wave propagation is performed using the [devito framework](https://www.devitoproject.org/), a Python package build to implement optimized stencil computation, capable of executing optimized computational kernels on several computer platforms, including CPUs, GPUs, and clusters thereof. Parallelization of shots across nodes is performed using [Ray](https://docs.ray.io/en/latest/), an open-source project which provides a simple and flexible API for building and running distributed applications. 


A [singularity definition file](https://github.com/fffarias/sfo-manuscript/blob/main/Dockerfile/Singularity.def) is provided to create a reproducible container and run this example properly, but you can also find a [requirements file](https://github.com/fffarias/sfo-manuscript/blob/main/requirements.txt) listing all of the project's dependencies and follow the instructions below to install them. In a nutshell, this are the full python packages required:

+ devito
+ ray
+ sfo

## Install dependencies  

To install [devito](https://www.devitoproject.org/) follow the instructions from [Devito documentation](https://www.devitoproject.org/devito/download.html). In case the best choice is to use a conda environment, the following steps should work as recommended on the [installation web page](https://www.devitoproject.org/devito/download.html#conda-environment) 
```
git clone https://github.com/devitocodes/devito.git
cd devito
conda env create -f environment-dev.yml
source activate devito
pip install -e .
```  

or it is also possible to install devito using pip installation, in this case simply type:
```
pip install devito
```  

The Ray package can be easily installed via `pip`. You can install the latest official version of Ray as follows.
```
pip install -U ray

```

The SFO optimizer can be used after cloning the original repository

```
https://github.com/Sohl-Dickstein/Sum-of-Functions-Optimizer
```
and setting the path where the ```sfo.py``` file is in the python sys
```
import sys
sys.path.append("./Sum-of-Functions-Optimizer")

```

Using the singularity definition file provided, the SFO repository is already cloned and added to the PYTHONPATH variable.

## Least-squares reverse time migration (LSRTM)

To provide subsurface images with better balanced amplitudes, higher resolution and fewer artifacts than standard migration, a least-sqaures migration should be considered. The LSRTM process involves, several wavefield computations of the Born modeling and its adjoint. To calculate these operators, I choose to use the `AcousticWaveSolver` Class from the Devito's folder `examples` or a Devito `Operator`, although it is also available on Devito, the operators needed to calculate the LSRTM in a medium with TTI anisotropy. In the python script available [here](https://github.com/fffarias/sfo-manuscript/blob/main/lsm.py), there are all the necessary steps to perform the LSRTM, since besides performing the forward linearized modeling and its adjoint, some previous actions need to be defined, such as creating an object that contains the velocity model and the acquisition geometry, for example. All these steps in different contexts are also explored in the [tutorials available](https://github.com/devitocodes/devito/tree/master/examples/seismic/tutorials) on the Devito section. Thus, the sequence adopted in the main function involves:

1. Creating velocity and reflectivity models associated with squared slowness. 
2. Defining the acquisition geometry.
3. Forward modeling for all the shots in parallel using ray, to generate the "observed data".
4. Running the SFO optimizer with the help of a function that returns objective function value and gradient for a subset of shots (batch size).



## Running on multiple CPUs or GPUs using Ray

Running LSRTM sequentially, even for a 2D example such as the Marmousi model, can be quite tedious, so given that there are resources available, the ideal would be to distribute the wavefield calculations across the available CPUs or GPUs. To accomplish this using Ray on a single machine, it is enough to start Ray by adding ```ray.init()``` to the code. By simply doing that, Ray will then be able to utilize all cores of your machine.

In order for the wave propagations to be performed on the GPU, you need to compile Devito in a slightly more sophisticated way following the [instructions provided](https://github.com/devitocodes/devito/wiki/Using-Devito-on-GPUs-with-NVIDIA-HPC-SDK), or use the appropriate [singularity recipe](https://github.com/fffarias/sfo-manuscript/blob/main/Dockerfile/Singularity_nvidia.def) to use Devito on GPUs with NVIDIA HPC SDK.

```
python lsm.py --bs=20
```

where ```bs``` controls the batch size. Other variables can be controlled from the command line:



| Symbol  | Description  |  
|---|---|
| so  | Discretization order of the spatial derivatives of the wave equation  |
| nbl  | Number of absorbing boundary points around the domain   |
| f0  | Source peak frequency   |
| tn  | Total simulation time   |
| nsrc  | Number of sources   |
| nrec  | Number of receivers    |
| npasses  | Number of passes through the entire data   |
| bs  | Batch size    |


As implemented, the output of the lsm.py script writes the inverted reflectivity to disk in a binary file and also generates a graph with the objective function values for each mini-batch.

## Special situations

If you have any question that you couldn't find the answer here, please email fernanda.farias8@gmail.com.

See also
------
 * Sum of Functions Optimizer arXiv paper: https://arxiv.org/abs/1311.2115
