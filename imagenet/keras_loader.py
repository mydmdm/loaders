"""preprocess the imagenet images and return the generators
requirements: 
pip install keras, opencv-python, scipy
"""
import os
import re
import numpy as np
import keras
import cv2


def image2ndarray(fpath: str):
    "read the image file and crop to (224,224,3) numpy array"
    idx = int(re.findall('ILSVRC2012_\w+_(\d+).JPEG', fpath)[-1])

    # Load (as BGR)
    img = cv2.imread(fpath)
    
    # Resize
    height, width, _ = img.shape
    new_height = height * 256 // min(img.shape[:2])
    new_width = width * 256 // min(img.shape[:2])
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Crop
    height, width, _ = img.shape
    startx = width//2 - (224//2)
    starty = height//2 - (224//2)
    img = img[starty:starty+224,startx:startx+224]
    assert img.shape[0] == 224 and img.shape[1] == 224, (img.shape, height, width)
    
    # Save (as RGB)
    return img[:,:,::-1], idx

class ILSVRC2012Gen(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(
        self, 
        img_folder: str, 
        devkit_t12: str, 
        synset_words: str='synset_words.txt',
        preprocess_fn: str='keras.applications.vgg19.preprocess_input',
        batch_size: int=32, 
        shuffle: bool=True
        ):
        'Initialization'
        self.list_IDs = [os.path.join(img_folder, x) for x in os.listdir(img_folder)]
        self.list_IDs.sort()
        self.batch_size = batch_size
        self.shuffle, self.indexes = shuffle, None
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)