import numpy as np
from math import sqrt
import timeit
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage.transform import resize
from imageio import imwrite
import os
import glob
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from pprint import pprint
import tensorflow as tf

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only use the first GPU
#   try:
#     tf.config.set_visible_devices(gpus[0], 'GPU')
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#   except RuntimeError as e:
#     # Visible devices must be set before GPUs have been initialized
#     print(e)


def print_tensors_in_ckpt(ckpt_path):
    print_tensors_in_checkpoint_file(
        ckpt_path, 
        tensor_name='', 
        all_tensors=False
    )
    
def load_variable_from_checkpoint(tensor_name, shape, ckpt_path, dtype=tf1.float64):
    tf1.reset_default_graph()

    # Load checkpoint from disk to compare phase shifts found by tensorflow
    tensor_graph = tf1.get_variable(tensor_name, shape=shape, dtype=dtype)
    with tf1.Session() as sess:
        saver = tf1.train.Saver([tensor_graph])
        saver.restore(sess, ckpt_path)
        
        tensor_value = sess.run(tensor_graph)

    return tensor_value

def get_last_checkpoint_folder(checkpoints_folder):
  checkpoints = glob.glob(checkpoints_folder + '*')
  latest_checkpoint_folder = max(checkpoints, key=os.path.getctime)
  return latest_checkpoint_folder 

CONFIG = type('', (), {})() 
CONFIG.checkpoint_path = get_last_checkpoint_folder('checkpoints/maskopt/quickdraw16_tiledpsf_4x/') + '/PhaseMask_ONN.ckpt-200000'
CONFIG.image_size = 192
CONFIG.scale = 4
CONFIG.phase_mask_size = CONFIG.image_size * CONFIG.scale
CONFIG.phase_mask_size_name = 'model/maskopt/phase_mask_height/height_map_sqrt'
CONFIG.wavelength = 550e-9
CONFIG.refraction_index = 1.509
CONFIG.phase_mask_heights_path = 'assets/quickdraw16_phase_mask_heights.npy'

print()
pprint(CONFIG.__dict__)
print()

print("Loaded phase mask heights from {}\n".format(CONFIG.checkpoint_path))
print_tensors_in_ckpt(CONFIG.checkpoint_path)

height_map_sqrt = load_variable_from_checkpoint(
    CONFIG.phase_mask_size_name, 
    [1, CONFIG.phase_mask_size, CONFIG.phase_mask_size, 1], 
    CONFIG.checkpoint_path, 
    dtype=tf.float64
)
height_map = np.square(height_map_sqrt.squeeze())        
    
print("min: {}, max: {}, mean: {}, std: {}".format(
    np.min(height_map), 
    np.max(height_map), 
    np.mean(height_map), 
    np.std(height_map)
))

np.save(CONFIG.phase_mask_heights_path, height_map)
print("Phase mask height path: ", CONFIG.phase_mask_heights_path)