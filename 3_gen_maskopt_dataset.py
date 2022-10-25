import os
from imageio import imwrite
import numpy as np

def imresize(im, size):
    return resize(im, size, order=3)

def imsave(path, im):
    imwrite(path, im)

# only need to run this to generate padded images for phase mask optimization
# generate training images for mask optimization
train_data = np.load('assets/quickdraw16_train.npy') # change to quickdraw16_train.npy after downloaded
if os.path.isdir('assets/quickdraw16_192'):
    print("Folder `assets/quickdraw16_192` exists")
else:
    os.mkdir('assets/quickdraw16_192')

for j in range(100): # change to 9999 or however many after train download
    i = np.random.randint(0, np.shape(train_data)[0])
    image = np.reshape(train_data[i], (28,28))
    
    # pad image
    pad_amt = 82 # 50 why not 64 or 80
    padded_image = np.pad(image, ((pad_amt, pad_amt)), 'constant', constant_values = (0,0))
    
    imsave('assets/quickdraw16_192/im_%04d.png' % (j), padded_image)