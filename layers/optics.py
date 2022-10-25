import abc

import tensorflow as tfn
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt

import fractions

# from layers.utils import fft2d, ifft2d, fftshift, ifftshift
def compl_exp_tf(phase, dtype=tf.complex128, name='complex_exp'):
    """Complex exponent via euler's formula, since Cuda doesn't have a GPU kernel for that.
    Casts to *dtype*.
    """
    phase = tf.cast(phase, tf.float64)
    return tf.add(
        tf.cast(tf.cos(phase), dtype=dtype),
        1.j * tf.cast(tf.sin(phase), dtype=dtype),
        name=name)

def phaseshifts_from_height_map(height_map, wave_lengths, refractive_index):
    '''Calculates the phase shifts created by a height map with certain
    refractive index for light with specific wave length.
    '''
    # refractive index difference
    delta_N = refractive_index - 1.000277
    # wave number
    wave_nos = 2. * np.pi / wave_lengths
    # phase delay indiced by height field
    phi = wave_nos * delta_N * height_map
    phase_shifts = compl_exp_tf(phi)
    return phase_shifts

class PhasePlate():
    def __init__(self,
                 wave_lengths,
                 height_map,
                 refractive_index,
                 height_tolerance=None,
                 lateral_tolerance=None):

        self.wave_lengths = wave_lengths
        self.height_map = height_map
        self.refractive_index = refractive_index
        self.height_tolerance = height_tolerance
        self.lateral_tolerance = lateral_tolerance

        # Add manufacturing tolerances in the form of height map noise
        if self.height_tolerance is not None:
            self.height_map += tf.random.uniform(shape=self.height_map.shape,
                                                 minval=-self.height_tolerance,
                                                 maxval=self.height_tolerance,
                                                 dtype=tf.float64)
            print("Phase plate with manufacturing tolerance %0.2e"%self.height_tolerance)

        self.phase_shifts = phaseshifts_from_height_map(
            self.height_map,
            self.wave_lengths,
            self.refractive_index
        )

    def __call__(self): 
        return self.phase_shifts

def height_map_element(
        map_shape,
        name,
        wave_lengths,
        block_size=1,
        height_map_initializer=None,
        height_map_regularizer=None,
        height_tolerance=None, # Default height tolerance is 2 nm.
        refractive_index=1.5
    ):

        b, h, w, c = map_shape
        input_shape = [b, h//block_size, w//block_size, c]

        if height_map_initializer is None:
            init_height_map_value = np.ones(shape=input_shape, dtype=np.float64) * 1e-4
            height_map_initializer = tf.compat.v1.constant_initializer(init_height_map_value)

        with tf.compat.v1.variable_scope(name, reuse=False):
            height_map_var = tf.compat.v1.get_variable(
                name="height_map_sqrt",
                shape=input_shape,
                dtype=tf.float64,
                trainable=True,
                initializer=height_map_initializer
            )
            height_map_full = tf.image.resize(
                height_map_var, 
                map_shape[1:3],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            height_map = tf.square(height_map_full, name='height_map')

            if height_map_regularizer is not None:
                tf.contrib.layers.apply_regularization(height_map_regularizer, weights_list=[height_map])

        element = PhasePlate(
            wave_lengths=wave_lengths,
            height_map=height_map,
            refractive_index=refractive_index,
            height_tolerance=height_tolerance
        )

        return element


class Sensor():
    def __init__(self, resolution, input_is_intensities=False):
        self.resolution = resolution
        self.input_is_intensities = input_is_intensities

    def __call__(self, input_field):
        if self.input_is_intensities:
            sensor_readings = input_field
        else:
            sensor_readings = tf.square(tf.abs(input_field))
        sensor_readings = tf.cast(
            sensor_readings, 
            tf.float64,
            name='sensor_readings'
        )
        sensor_readings = area_downsampling_tf(sensor_readings, self.resolution[0])

        return sensor_readings