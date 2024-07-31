"""
Code for creating grid sampled circular harmonic filters
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

# VARIABLES FOR RADIAL PROFILE AND PHASE OFFSET

def get_weights_dict(layer: keras.Layer, shape: list, max_order: int | tuple, in_max_order: int | tuple,
                     conjugate_weights: bool = False, initializer: keras.Initializer = None,
                     regularizer: keras.Regularizer = None, n_rings: int | None = None, name: str ='W'
                     ) -> dict[int, tf.Variable]:
    """Creates a dictionary containing randomly intitialized weights (radial profile R) for every rotation order. 
    Each variable is independent from another and has the shape [n_rings, in_channels, out_channels]. 
    This means weights are NOT shared across rotation order, input channel or filter.

    Args:
        layer (Layer): The HConvLayer to add the weights to.
        shape (list): The shape of the filter: [h, w, in_channels, out_channels]
        max_order (int | tuple): The maximum outgoing feature map rotation order if starting at 0
            or the range of rotation orders if not starting at 0.
        in_max_order (int | tuple): The maximum ingoing feature map rotation order if starting at 0
            or the range of rotation orders if not starting at 0.
        conjugate_weights (bool, optional): Whether to use conjugate weights for negative orders. Defaults to False.
        initializer (Initializer, optional): Initializer used to initialize the radial profile. Defaults to None.
        regularizer (Regularizer, optional): The regularzation applied to the radial profile. Defaults to None.
        n_rings (int | None, optional): The number of magnitudes with distinct R values. Defaults to one ring
            per filter unit and at least 2.
        name (str, optional): The name of the tensorflow variables. Defaults to 'W'.

    Returns:
        dict[int, tf.Variable]: A dictionary containing a tensorflow variable with randomly intitialized weights 
            (R(r) values) for every rotation order. Each variable is independent from another and has the 
            shape [n_rings, in_channels, out_channels]. This means weights are NOT shared across rotation 
            order, input channel or filter.
    """
    # compute required filter rotation orders
    if isinstance(max_order, int): max_order = (0, max_order)
    if isinstance(in_max_order, int): in_max_order = (0, in_max_order)
    m_min, m_max = max_order[0]-in_max_order[1], max_order[1]-in_max_order[0]
    if conjugate_weights:
        m_min, m_max = 0, max(abs(m_min), abs(m_max))
    orders = range(m_min, m_max+1)
        
    weights_dict = {} # dict with weight tensor for each filter rotation order
    for m in orders: # seperate learnabel parameters for each rotation order m
        if n_rings is None: # the number of magnitudes with distinct R values
            n_rings = np.maximum(int(shape[0]/2), 2) # one ring for every unit in the filter, but at least 2
        sh = [n_rings,] + shape[2:] # Separate R value for each ring x input channel x filter
        weights_dict[m] = layer.add_weight(name=name + '_' + str(m), dtype=tf.float32, shape=sh,
                                           initializer=initializer, regularizer=regularizer)
    return weights_dict


def get_phase_dict(layer: keras.Layer, n_in: int, n_out: int, max_order: int | tuple, in_max_order: int | tuple,
                   conjugate_weights: bool = False, initializer: keras.Initializer = None, name: str ='b'
                   ) -> dict[int, tf.Variable]:
    """Creates a dictionary containing phase offsets (0-2pi) for every rotation order. Each variable is independent 
    from another and has the shape [1, 1, n_in, n_out]. This means phase offsets are NOT shared across rotation order, 
    input channel or filter.

    Args:
        layer (Layer): The HConvLayer to add the weights to.
        n_in (int): The number of input channels.
        n_out (int): The number of filters (output channels)
        max_order (int | tuple): The maximum outgoing feature map rotation order if starting at 0
            or the range of rotation orders if not starting at 0.
        in_max_order (int | tuple): The maximum ingoing feature map rotation order if starting at 0
            or the range of rotation orders if not starting at 0.
        conjugate_weights (bool, optional): Whether to use conjugate weights for negative orders. Defaults to False.
        initializer (Initializer, optional): Initializer used to initialize the phase offset. Defaults to None.
        name (str, optional): The name of the tensorflow variables. Defaults to 'b'.

    Returns:
        dict[int, tf.Variable]: A dictionary containing a tensorflow variable with phase offsets (0-2pi) for every 
        rotation order. Each variable is independent from another and has the shape [1, 1, in_channels, out_channels]. 
        This means phase offsets are NOT shared across rotation order, input channel or filter.
    """
    # compute required filter rotation orders
    if isinstance(max_order, int): max_order = (0, max_order)
    if isinstance(in_max_order, int): in_max_order = (0, in_max_order)
    m_min, m_max = max_order[0]-in_max_order[1], max_order[1]-in_max_order[0]
    if conjugate_weights:
        m_min, m_max = 0, max(abs(m_min), abs(m_max))
    orders = range(m_min, m_max+1)
        
    phase_dict = {} # dict with phase tensor for each filter rotation order
    for m in orders:
        # random initialized phase offsets
        # for each rotation order x input channel x filter separately
        phase = layer.add_weight(name=name+'_'+str(m), dtype=tf.float32,
                                shape=[1,1,n_in,n_out], initializer=initializer) # transform into variables
        phase_dict[m] = phase
    return phase_dict


# GRID SAMPLING

def get_interpolation_weights(filter_size: int, m: int, n_rings: int | None = None) -> np.ndarray:
    """Calculates gaussian weights based on distance for resampling the ring filter on a grid.

    Args:
        filter_size (int): The size of the filter.
        m (int): The rotation order of the filter.
        n_rings (int | None, optional): The number of distinct magnitudes with distinct R values. Defaults to one ring
            per filter unit and at least 2.

    Returns:
        np.ndarray: Weights as an array with shape [n_rings, angles, positions], where n_ring x angle define the position 
            in the ring and position the position in the grid.
    """
    if n_rings is None: # the number of radii with distinct R values
        n_rings = np.maximum(int(filter_size/2), 2) # one ring for every unit in the filter, but at least 2
    n_rings = np.int64(n_rings)
    # the distances of the rings to the origin
    # starting at 0 for m=0 and 1 for m>0
    radii = tf.linspace(int(m!=0), n_rings-0.5, n_rings)

    # computing the polar locations
    angular_samples = n_samples(filter_size) # number of angles to sample
    # the angles to sample from (uniform distribution over 0-2pi)
    lin = tf.multiply(tf.cast(2*np.pi/angular_samples, dtype=tf.float64), tf.range(0, angular_samples, dtype=tf.float64))
    
    # the y,x positions of the angles on the unit circle
    circle_locations = tf.stack([-tf.math.sin(lin), tf.math.cos(lin)])

    radii = radii[:,tf.newaxis,tf.newaxis,tf.newaxis] # shape=[n_rings,1,1,1]
    circle_locations = circle_locations[tf.newaxis,:,:,tf.newaxis] # shape=[1,y/x,angle,1]
    polar_locations = radii*circle_locations # y,x positions of the polar locations, shape=[n_rings,y/x,angle,1]
    
    # computing the grid locations
    filter_center = tf.constant([filter_size/2, filter_size/2], dtype=tf.float64)
    grid_locations = L2_grid(filter_center, filter_size) # the y,x positions in the grid filter
    grid_locations = grid_locations[tf.newaxis,:,tf.newaxis,:] # [y/x,pos] -> [1,y/x,1,pos]

    # create gaussian weightening depending on the distance between sample position and polar locations
    diff = polar_locations - grid_locations # shape=[n_rings,y/x,angle,pos]
    distance_squared = tf.reduce_sum(diff**2, axis=1) # shape=[n_rings, angle, pos]
    bandwidth = 0.5
    weights = tf.math.exp(-0.5*distance_squared/(bandwidth**2))
    
    # normalize weights
    return weights/tf.reduce_sum(weights, axis=2, keepdims=True)


def get_filters(r: dict[int, tf.Variable], filter_size: int, p: dict[int, tf.Variable] | None = None, 
                n_rings: int | None = None) -> dict[int, tuple[tf.Variable, tf.Variable]]:
    """Samles the polar filters on a grid using gaussian resampling.

    Args:
        r (dict[int, tf.Variable]): A dictionary containing a tensorflow variable with radial profile
            for every rotation order.
        filter_size (int): The size of the filter.
        p (dict[int, tf.Variable] | None, optional): A dictionary containing a tensorflow variable with phase 
            offsets for every rotation order. Defaults to None.
        n_rings (int | None, optional): The number of rings with distinct R values. Defaults to None.

    Returns:
        dict[int, tuple[tf.Variable, tf.Variable]]: Dictionary containing the real and imaginary parts of the grid 
            filters for the different rotation orders. Real and imaginary parts have both the shape 
            [filter_size, filter_size, in_channels, filters].
    """
    filters = {} # for the grid filters of each rotation order
    angular_samples = n_samples(filter_size) # number of angular samples on each ring
    
    from scipy.linalg import dft

    # iterating over all rotation orders
    for m, r in r.items():
        rsh = r.shape.as_list() # [rings,in_channels,filters]
        
        # weightening up the phase terms on a ring for every grid position
        weights = get_interpolation_weights(filter_size, m, n_rings=n_rings) # shape=[n_rings,angles,gridpositions]
        DFT = tf.constant(dft(angular_samples)[m,:])
        phases = tf.pow(DFT, -1) # vector containing the e^{i*m* (2pi*j/n)} terms (shape=[angles])
        weights = tf.cast(weights, dtype=tf.complex128)
        weighted_phases = tf.transpose(tf.tensordot(phases, weights, axes = [[0], [1]]))
        # shape=T([angles]x[rings,angles,gridpositions])=T([rings,gridpositions])=[gridpositions,rings]

        # splitting in real and imaginary part
        real = tf.cast(tf.math.real(weighted_phases), dtype=tf.float32) # shape=[gridpositions,rings]
        imag = tf.cast(tf.math.imag(weighted_phases), dtype=tf.float32) # shape=[gridpositions,rings]
        
        # summing over the rings for alle positions
        r = tf.reshape(r, tf.stack([rsh[0],rsh[1]*rsh[2]])) # shape=[rings,inputchannels*filters]
        ureal = tf.reshape(tf.matmul(real, r), tf.stack([filter_size, filter_size, rsh[1], rsh[2]]))
        uimag = tf.reshape(tf.matmul(imag, r), tf.stack([filter_size, filter_size, rsh[1], rsh[2]])) 
        # shape=[k,k,inputchannels,filters]
        
        # Sampled(gridpos, c, f) = sum_(ring) sum_(angle) R_c,f(ring) * e^{im * 2pi*angle/n} * weights[ring,angle,gridpos]
        #                        = sum_(ring) R_c,f(ring) * sum_(angle) phase[angle] * weights[ring,angle,pos]
        #                        = sum_(ring) R_c,f(ring) * weighted_phases[pos,ring]
        #                        = sum_(ring) r[ring,c,f] * weighted_phases[pos,ring]
        #                        = ureal[gridpos,c,f] + i*uimag[gridpos,c,f]
        
        if p is not None:
            # Adds phase offset by rotating each coordinate
            ureal_ = tf.cos(p[m])*ureal + tf.sin(p[m])*uimag
            uimag = -tf.sin(p[m])*ureal + tf.cos(p[m])*uimag
            ureal = ureal_
            
        filters[m] = (ureal, uimag)
        
    return filters


def n_samples(filter_size: int) -> int:
    """Calculates the number of angles to sample from on the rings.
    This is a sample for each unit on the most outer ring, but at least 101 times.

    Args:
        filter_size (int): The size of the filter.

    Returns:
        int: The number of angles to sample from on the rings.
    """
    return np.maximum(np.ceil(np.pi*filter_size),101) ############## <--- One source of instability


def L2_grid(center: np.ndarray[float], size: int) -> np.ndarray:
    """Creates a coordinate system for a filter of a certain size, which is centered at the origin.
    The coordinate system is returned as two stacked arrays. The first contains the x-coordinates 
    and the latter the y-coordinates.
    
    Args:
        center (np.ndarray[float]): The (x,y) coordinates of the filter center
        size (int): The size of the filter.

    Returns:
        np.ndarray: The coordinates of the origin centered filter as two stacked arrays. 
            The first contains the x-coordinates and the latter the y-coordinates.
    """
    # Get neighbourhoods
    lin = lin = tf.add(tf.range(0, size, dtype=tf.float64), 0.5) # [0.5, 1.5, 2.5] for size=3
    j, i = tf.meshgrid(lin, lin) # 2D coordinate system in the size of the filter
    
    # centered around origin
    i = i - center[1] 
    j = j - center[0]
    
    return tf.stack((tf.reshape(i, [-1]), tf.reshape(j, [-1]))) # flattens coordinates