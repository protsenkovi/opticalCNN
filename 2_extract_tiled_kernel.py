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

def get_last_checkpoint_folder(checkpoints_folder):
  checkpoints = glob.glob(checkpoints_folder + '*')
  latest_checkpoint_folder = max(checkpoints, key=os.path.getctime)
  return latest_checkpoint_folder 

CONFIG = type('', (), {})() 
CONFIG.checkpoint_path = get_last_checkpoint_folder('checkpoints/quickdraw_tiled_nonneg/') + '/model.ckpt-10000'
CONFIG.kernel_pad_amount = 4
CONFIG.kernel_count = 16
CONFIG.kernel_size = 32
CONFIG.spatial_kernel_path = 'assets/quickdraw16_tiledpsf.npy'


print()
pprint(CONFIG.__dict__)
print()

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


print("Loaded classifier weights from {}\n".format(CONFIG.checkpoint_path))
print_tensors_in_ckpt(CONFIG.checkpoint_path)

# load tiled PSF
# make sure these dimensions match the variable size printed above
# this will depend on how you chose to kernel size during PSF optimization
kernels = np.zeros((CONFIG.kernel_size, CONFIG.kernel_size, CONFIG.kernel_count))
count = 0
N = int(sqrt(CONFIG.kernel_count))
for i in range(N):
    for j in range(N):
        varname = 'h_conv1/kernel_' + str(i) + str(j)
        var = load_variable_from_checkpoint(
          varname, 
          [CONFIG.kernel_size, CONFIG.kernel_size,1,1], 
          CONFIG.checkpoint_path, 
          dtype=tf1.float32
        )
        kernels[:,:,count] = np.abs(var.squeeze())
        count += 1
        
print("\nCreating 4x4 spatial kernel of 16 convolutional kernels")

# tile individual kernels on one large PSF
kernels_paddings = [
  [CONFIG.kernel_pad_amount, CONFIG.kernel_pad_amount], 
  [CONFIG.kernel_pad_amount, CONFIG.kernel_pad_amount]
]
padded_kernels = []
for i in range(CONFIG.kernel_count):
  padded_kernels.append(np.pad(kernels[:,:,i], kernels_paddings, mode='constant'))

tmp = []
for i in range(N):
  tmp.append(np.concatenate(padded_kernels[i*N:i*N+CONFIG.kernel_pad_amount], axis=1))
psf = np.concatenate(tmp, axis=0)

print("\nSpatial kernel shape: ", np.shape(psf))
print()

# save PSF for phase mask optimization
# can choose to pad this before saving as .npy for phase mask opt
np.save(CONFIG.spatial_kernel_path, psf)
print("Spatial kernel path: ", CONFIG.spatial_kernel_path)