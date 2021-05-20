import numpy as np
import ray
import scipy.ndimage
from argparse import ArgumentParser
from examples.seismic import Model
from examples.seismic import AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import Receiver
from devito import Function
import matplotlib.pyplot as plt
from sfo import SFO


ray.init(num_cpus=20)

# Serial FWI objective function
@ray.remote(num_returns=2)
def f_df_single_shot(s, d_obs, ixsrc, geometry, model, space_order): 

    # Geometry for current shot
    geometry_i = AcquisitionGeometry(model, geometry.rec_positions, geometry.src_positions[ixsrc,:],
                                     geometry.t0, geometry.tn, f0=geometry.f0, src_type=geometry.src_type)

    #set reflectivity
    dm = np.zeros(model.grid.shape, dtype=np.float32)
    dm[model.nbl:-model.nbl,model.nbl:-model.nbl] = s[:].reshape((model.shape[0],model.shape[1]))                                  

    # Devito objects for gradient and data residual
    grad = Function(name="grad", grid=model.grid)

    residual = Receiver(name='rec', grid=model.grid,
                        time_range=geometry_i.time_axis, 
                        coordinates=geometry_i.rec_positions)

    solver = AcousticWaveSolver(model, geometry_i, space_order=space_order)

    # Predicted data and residual
    _, u0, _ = solver.forward(save=True)
    d_pred = solver.born(dm)[0]
    residual.data[:] = d_pred.data[:] - d_obs[:]
    fval = .5*np.linalg.norm(residual.data.flatten())**2

    # Function value and gradient        
    solver.gradient(rec=residual, u=u0, vp=model.vp, grad=grad)

    # Convert to numpy array and remove absorbing boundaries
    grad_crop = np.array(grad.data[:],dtype=np.float32)[model.nbl:-model.nbl, model.nbl:-model.nbl]

    # Do not update water layer
    grad_crop[:,:36] = 0.

    return fval, grad_crop.flatten()

# Parallel FWI objective function
def f_df_multi_shots(s, data_list, geometry, model, space_order):

    nsrc_batch = len(data_list[1])
    d_obs = data_list[0].reshape((geometry.nt,geometry.nrec,nsrc_batch))
    
    future = []
    for i in range(nsrc_batch):        
        args = [s, d_obs[:,:,i], data_list[1][i], geometry, model, space_order]                
        future.append(f_df_single_shot.remote(*args))

    fval = 0.0
    grad = np.zeros(model.shape[0]*model.shape[1])
    for i in range(nsrc_batch):
        fval = fval + ray.get(future[i][0])
        grad = grad + ray.get(future[i][1])
        
    
    return fval, grad.flatten()

def get_true_model(shape, spacing, origin, nbl, space_order, **kwargs):
    ''' Read Vp file and set to Model object. (km/s)
    '''
    data_path = "data/vel.bin"
    vp = 1e-3 * np.fromfile(data_path, dtype='float32', sep="")
    vp = vp.reshape(shape)
    return Model(space_order=space_order, vp=vp, origin=origin, shape=shape,
                 dtype=np.float32, spacing=spacing, nbl=nbl, bcs="damp")

def get_smooth_model(shape, spacing, origin, nbl, space_order, **kwargs):
    ''' Read Vp file apply smoothing filter and set it to Model 
        object. (km/s)
    '''
    data_path = "data/vel.bin"
    vp = 1e-3 * np.fromfile(data_path, dtype='float32', sep="")
    vp = vp.reshape(shape)
    v0 = scipy.ndimage.gaussian_filter(vp, sigma=10)    
    v0[:,:36] = 1.5 # Do not smooth water layer
    return Model(space_order=space_order, vp=v0, origin=origin, shape=shape,
                 dtype=np.float32, spacing=spacing, nbl=nbl, bcs="damp")

def set_geometry(model, nsrc, nrec, f0, tn, t0=0):  

    # First, position source centrally in all dimensions, then set depth
    src_coordinates = np.empty((nsrc, 2))
    src_coordinates[:, 0] = np.linspace(0, model.domain_size[0], num=nsrc)
    src_coordinates[:, 1] = 20.  # Depth is 20m

    # Define acquisition geometry: receivers

    # Initialize receivers for synthetic and imaging data
    rec_coordinates = np.empty((nrec, 2))
    rec_coordinates[:, 0] = np.linspace(0, model.domain_size[0], num=nrec)
    rec_coordinates[:, 1] = 20.

    # Geometry
    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')

    return geometry

@ray.remote
def forward_modeling(dm, isrc, model, geometry, space_order):
    geometry_i = AcquisitionGeometry(model, geometry.rec_positions, geometry.src_positions[isrc,:],
                                     geometry.t0, geometry.tn, f0=geometry.f0, src_type=geometry.src_type)    
    solver = AcousticWaveSolver(model, geometry_i, space_order=space_order)
    d_obs, _, _, _ = solver.born(dm)    
    
    return (d_obs.resample(geometry.dt).data[:][0:geometry.nt, :]).flatten()

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[1]    
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)                
        yield inputs[excerpt], targets[:,excerpt]

# Write subfunctions as list
def set_subreferences(dobs, geometry, batch_size):
    
    sub_refs = []
    for batch in iterate_minibatches(np.array(range(geometry.nsrc)), dobs, batch_size, shuffle=True):
            x_batch, dobs_batch = batch
            sub_refs.append([dobs_batch,x_batch])

    return sub_refs

def main(shape, spacing, origin, nbl, space_order,
         xs, xr, tn, f0, npasses, batch_size, **kwargs):    

    # Get true model
    true_model = get_true_model(shape, spacing, origin, nbl, space_order)

    # Get smooth model
    smooth_model = get_smooth_model(shape, spacing, origin, nbl, space_order)
        
    # Compute initial born perturbation from m - m0
    dm = (true_model.vp.data**(-2) - smooth_model.vp.data**(-2))

    # Geometry
    nsrc = xs.shape[0]
    nrec = xr.shape[0]
    geometry0 = set_geometry(smooth_model, nsrc, nrec, f0, tn, t0=0)

    # Compute observed data in parallel (inverse crime). 
    # In real life we would read the SEG-Y data here.    
    futures = []
    for i in range(geometry0.nsrc):
        args = [dm, i, smooth_model, geometry0, space_order]
        futures.append(forward_modeling.remote(*args))
    
    dobs = np.zeros((geometry0.nt*geometry0.nrec,geometry0.nsrc),dtype=np.float32)    
    for i in range(geometry0.nsrc):
        dobs[:,i] = ray.get(futures[i])  
    
    # List containing an identifying element for each subfunction
    sub_refs = set_subreferences(dobs, geometry0, batch_size)            
           
    # Initial guess
    theta_init = np.zeros(smooth_model.shape, dtype=np.float32)               
      
    # # initialize the optimizer
    optimizer = SFO(f_df_multi_shots, theta_init, sub_refs, [geometry0, smooth_model, space_order]) 

    # # run the optimizer for npasses pass through the data
    theta = optimizer.optimize(num_passes=npasses) 

    # Write inverted reflectivity to disk    
    file = open('output/dvel-final.bin', "wb")
    scopy = theta.reshape(smooth_model.shape).astype(np.float32).copy(order='C')
    file.write(scopy)  
    
    plt.plot(np.array(optimizer.hist_f_flat))
    plt.xlabel('Iteration')
    plt.ylabel('Minibatch Function Value')
    plt.title('Convergence Trace')  
    plt.savefig('output/history_sfo.png')
    
    

if __name__ == '__main__':   
    description = ("Script for LSM example using devito")
    parser = ArgumentParser(description=description)
   
    parser.add_argument("--so", default=6,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbl", default=30,
                        type=int, help="Number of boundary layers around the domain")
    parser.add_argument("--f0", default=0.015,
                        type=float, help="Source peak frequency")  
    parser.add_argument("--tn", default=5000.0,
                        type=float, help="Total simulation time")  
    parser.add_argument("--nsrc", default=680,
                        type=int, help="Number of sources")
    parser.add_argument("--nrec", default=1360,
                        type=int, help="Number of receivers") 
    parser.add_argument("--npasses", default=1,
                        type=int, help="Number of passes through the data") 
    parser.add_argument("--bs", default=20,
                        type=int, help="Batch size")  
    
    args = parser.parse_args()

    #model parameter
    shape = (1360, 280)
    spacing = (12.5, 12.5)
    origin = (0, 0)

    #geometry arrange
    xs = np.linspace(0, shape[0]*spacing[0] + origin[0], num=args.nsrc)    
    xr = np.linspace(0, shape[0]*spacing[0] + origin[0], num=args.nrec)

    main(shape=shape, spacing=spacing, origin=origin, xs=xs, xr=xr, 
         nbl=args.nbl, space_order=args.so, tn=args.tn, 
         f0=args.f0, npasses=args.npasses, batch_size=args.bs)





