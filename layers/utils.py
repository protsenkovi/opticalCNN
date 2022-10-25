import abc

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import fractions

from layers.optics import height_map_element

def fft2d(a_tensor, dtype=tf.complex128):
    """Takes images of shape [batch_size, x, y, channels] and transposes them
    correctly for tensorflows fft2d to work.
    """
    # Tensorflow's fft only supports complex64 dtype
    a_tensor = tf.cast(a_tensor, tf.complex64)
    # Tensorflow's FFT operates on the two innermost (last two!) dimensions
    a_tensor_transp = tf.transpose(a=a_tensor, perm=[0,3,1,2])
    # ACTUAL FFT
    a_fft2d = tf.signal.fft2d(a_tensor_transp)
    a_fft2d = tf.cast(a_fft2d, dtype)
    a_fft2d = tf.transpose(a=a_fft2d, perm=[0,2,3,1])
    return a_fft2d

def ifft2d(a_tensor, dtype=tf.complex128):
    a_tensor = tf.transpose(a=a_tensor, perm=[0,3,1,2])
    a_tensor = tf.cast(a_tensor, tf.complex64)
    # ACTUAL FFT
    a_ifft2d_transp = tf.signal.ifft2d(a_tensor)
    # Transpose back to [batch_size, x, y, channels]
    a_ifft2d = tf.transpose(a=a_ifft2d_transp, perm=[0,2,3,1])
    a_ifft2d = tf.cast(a_ifft2d, dtype)
    return a_ifft2d

def fftshift(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(1, 3):
        split = (input_shape[axis] + 1)//2
        mylist = np.concatenate((np.arange(split, input_shape[axis]), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor

def fftshift_psf(a_tensor):
    """filter shape is (height, width, in_channels, out_channels)"""
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(0, 2):
        split = (input_shape[axis] + 1)//2
        mylist = np.concatenate((np.arange(split, input_shape[axis]), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor

def ifftshift(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(1, 3):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        mylist = np.concatenate((np.arange(split, n), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor

def psf2otf(input_filter, output_size):
    """Convert 4D tensorflow filter into its FFT.
    """
    # pad out to output_size with zeros
    # circularly shift so center pixel is at 0,0

    shifted = fftshift_psf(input_filter)

    tmp = tf.transpose(a=shifted, perm=[2,3,0,1])
    # ACTUAL FFT
    tmp = tf.signal.fft2d(tf.complex(tmp, 0.))
    tmp = tf.transpose(a=tmp, perm=[2,3,0,1])
    return tmp

def fft_conv2d(img, psf):
    """Implements convolution in the frequency domain, with circular boundary conditions.
    Args:
        img: image with shape (batch_size, height, width, num_img_channels)
        psf: filters with shape (kernel_height, kernel_width, num_img_channels, num_filters)
    """
    img = tf.cast(img, dtype=tf.float32)
    psf = tf.cast(psf, dtype=tf.float32)
    img_shape = img.shape.as_list()
    img_side_size = img_shape[1]
    psf_side_size = psf.shape.as_list()[0]

    img_pad_amount = int(img_side_size/2)
    target_side_size = img_side_size + 2*img_pad_amount
    psf_pad_amount = (target_side_size - psf_side_size)//2

    img = fftpad(img, img_pad_amount)
    psf = fftpad_psf(psf, psf_pad_amount)
    img_shape = img.shape.as_list()

    img_fft = fft2d(img)
    img_fft = tf.cast(img_fft, tf.complex64) 

    otf = psf2otf(psf, output_size=img_shape[1:3])
    otf = tf.transpose(a=otf, perm=[2,0,1,3])
    otf = tf.cast(otf, tf.complex64)
    result = ifft2d(img_fft * otf)
    
    result = tf.math.real(result)
    result = tf.cast(result, tf.float32)

    result = fftunpad(result, img_pad_amount)

    return result

def fft_conv2d_adjoint(img, psf):
    """Implements convolution in the frequency domain, with circular boundary conditions.
    Args:
        img: image with shape (batch_size, height, width, num_img_channels)
        psf: filters with shape (kernel_height, kernel_width, num_img_channels, num_filters)
    """
    img = tf.cast(img, dtype=tf.float32)
    psf = tf.cast(psf, dtype=tf.float32)
    img_shape = img.shape.as_list()
    img_side_size = img_shape[1]
    psf_side_size = psf.shape.as_list()[0]

    img_pad_amount = int(img_side_size/2)
    target_side_size = img_side_size + 2*img_pad_amount
    psf_pad_amount = (target_side_size - psf_side_size)//2

    img = fftpad(img, img_pad_amount)
    psf = fftpad_psf(psf, psf_pad_amount)
    img_shape = img.shape.as_list()

    img_fft = fft2d(img)
    img_fft = tf.cast(img_fft, tf.complex64) 

    otf = psf2otf(psf, output_size=img_shape[1:3])
    otf = tf.transpose(a=otf, perm=[2,0,1,3])
    otf = tf.cast(otf, tf.complex64)
    result = ifft2d(img_fft * tf.math.conj(otf))
    
    result = tf.math.real(result)
    result = tf.cast(result, tf.float32)

    result = fftunpad(result, img_pad_amount)

    return result

# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape, name=None):
  """Create a weight variable with appropriate initialization."""
  # initial = tf.truncated_normal(shape, stddev=0.1)
  # return tf.Variable(initial, name=name) 
  return tf.compat.v1.get_variable(name, shape, initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
  
def bias_variable(shape, name=None):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def conv2d(x, W):
  return tf.nn.conv2d(input=x, filters=W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool2d(
    input=x, 
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1], 
    padding='SAME'
  )

def shifted_relu(x):
    shift = tf.reduce_mean(input_tensor=x)
    return tf.nn.relu(x - shift) + shift

def cycle_W_conv(W_conv, din):
    # cycle through conv kernels
    w_list = []
    for i in range(din):
        w_list.append(tf.concat([W_conv[:,:,:,i:], W_conv[:,:,:,:i]], axis=3))
    return tf.concat(w_list, axis=2)
        
def fftpad(x, padamt):
    """Add padding before convolution with FFT"""
    paddings = tf.constant([[0, 0,], [padamt, padamt], [padamt, padamt], [0, 0]])
    return tf.pad(tensor=x, paddings=paddings)

def fftpad_psf(x, padamt):
    """Add padding before convolution with FFT""" 
    paddings = tf.constant([[padamt, padamt,], [padamt, padamt], [0, 0], [0, 0]]) #[x, y, 1, 1]
    return tf.pad(tensor=x, paddings=paddings)

def fftunpad(x, unpadamt):
    """Remove padding after convolution with FFT"""
    return x[:,unpadamt:-unpadamt, unpadamt:-unpadamt, :]

##############################
# NN layers
##############################

def optical_conv_layer(
    input_field, 
    refraction_index, 
    wavelength, 
    name = 'optical_conv'
):
    dims = input_field.get_shape().as_list()
    with tf.compat.v1.variable_scope(name):
        initializer = tf.compat.v1.random_uniform_initializer(minval=0.999e-4, maxval=1.001e-4)

        mask = height_map_element(
            [1, dims[1], dims[2], 1],
            wave_lengths = wavelength,
            refractive_index = refraction_index,
            height_map_initializer = initializer,
            name = 'phase_mask_height'
        )

        atf = mask()
        psfc = fftshift(ifft2d(ifftshift(atf)))
        psf = tf.square(tf.abs(psfc))   # amplitude to intensities
        psf /= tf.reduce_sum(psf)       # conservation of energy why total! keep_dims=True, dims=[1,2,3]
        psf = tf.cast(psf, tf.float32)

        # not coherent
        psf = tf.expand_dims(tf.expand_dims(tf.squeeze(psf), -1), -1)
  
        output_img = fft_conv2d(input_field, psf) # should be adjoint?

        output_img = tf.abs(output_img)
            
        # print("output_img shape {}".format(output_img.shape))

        return output_img

def least_common_multiple(a, b):
    return abs(a * b) / fractions.gcd(a,b) if a and b else 0

def area_downsampling_tf(input_image, target_side_length):
    input_shape = input_image.shape.as_list()
    input_image = tf.cast(input_image, tf.float32)
    # import sys
    if not input_shape[1] % target_side_length:
        factor = int(input_shape[1]/target_side_length)
        print("INPUT SHAPE {} TARGET SIDE LENGTH {}".format(input_shape, target_side_length))
        # output_image = tf.nn.avg_pool(
        #     input_image,
        #     ksize=[1, factor, factor, 1],
        #     strides=[1, factor, factor, 1],
        #     padding="SAME", data_format="NHWC")
        
        output_image = input_image
    else:
        # We upsample the image and then average pool
        lcm_factor = least_common_multiple(target_side_length, input_shape[1]) / target_side_length
        print("LCM FACTOR: {}".format(lcm_factor))
        if lcm_factor>10:
            print("Warning: area downsampling is very expensive and not precise if source and target wave length have a large least common multiple")
            upsample_factor=10
        else:
            upsample_factor = int(lcm_factor)

        img_upsampled = tf.image.resize(
            input_image,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, 
            size=2*[upsample_factor*target_side_length]
        )
        output_image = tf.nn.avg_pool(
            img_upsampled,
            ksize = [1, upsample_factor, upsample_factor, 1],
            strides = [1, upsample_factor, upsample_factor, 1],
            padding = "SAME")
    # sys.exit()
    return output_image


def get_intensities(input_field):
    return tf.square(tf.abs(input_field), name='intensities')

def gaussian_noise(image, stddev=0.001):
    dtype = image.dtype
    return image + tf.random.normal(image.shape, 0.0, stddev,dtype=dtype)