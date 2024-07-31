import numpy as np
import tensorflow as tf


def transform_filter_2d_nhwc(w: tf.Variable, flat_indices: np.ndarray, shape_info: tuple) -> tf.Variable:
    """Transform a set of filters defined on a split plane group G.
    This is the first step of the G-Conv. The user will typically not have to call this function directly.

    Args:
        w (tf.Variable): The input filter bank w has shape [n, n, ni * nti, no], where:
            n: the filter width and height
            ni: the number of input channels (note: the input feature map is assumed to have ni * nti number of channels)
            nti: the number of transformations in H (the stabilizer of the origin in the input space)
            For example, nti == 1 for images / functions on Z2, since only the identity translation leaves the origin invariant.
            Similarly, nti == 4 for the group p4, because there are 4 transformations in p4 (namely, the four rotations around
            the origin) that leave the origin in p4 (i.e. the identity transformation) fixed.
            no: the number of output channels (note: the G-Conv will actually create no * nto number of channels, see below).
        flat_indices (np.ndarray): The index array has shape [nto, nti, n, n] and maps [nto, nti, n, n] to the corresponding index.
        shape_info (tuple): Shape information of the gconv layer. Can be obtained by `splitgconv2d.gconv2d_util`

    Returns:
        tf.Variable: The output filter bank transformed_w has shape (n, n, ni * nti, nto * no),
        so there are nto times as many filters in the output as we had in the input w.
    """

    # The indexing is done using tf.gather. This function can only do integer indexing along the first axis.
    # We want to index the spatial and transformation axes of our filter, so we must flatten them into one axis.
    no, nto, ni, nti, n = shape_info
    w_flat = tf.reshape(w, [n * n * nti, ni, no]) # shape (n * n * nti, ni, no)
    
    # Do the transformation / indexing operation.
    # flat_indices[filter_rot,input_rot,upos,vpos] -> index = input_rot' + nti*vpos' + nti*n*upos'
    transformed_w = tf.gather(w_flat, flat_indices)              # shape (nto, nti, n, n, ni, no)

    # Put the axes in the right order, and collapse them to get a standard shape filter bank
    transformed_w = tf.transpose(transformed_w, [2, 3, 1, 4, 0, 5])           # shape (n, n, nti, ni, nto, nt)
    transformed_w = tf.reshape(transformed_w, [n, n, nti * ni, nto * no])     # shape (n, n, nti * ni, nto * no)

    return transformed_w