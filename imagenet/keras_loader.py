"""preprocess the imagenet images and return the generators
requirements: 
pip install keras tensorflow opencv-python pandas
"""
import os
import re
import numpy as np
from keras.utils import Sequence
import cv2
from scipy.io import loadmat
import pandas as pd


from time import time

def timer(func):
    def wraper(*args, **kwargs):
        before = time()
        result = func(*args, **kwargs)
        after = time()
        print(f"timer <{func.__name__}>: {(after - before)*1e3}ms")
        return result
    return wraper


def pandas_find_rows(df: pd.DataFrame, colname: str, colval):
    return df.loc [df[colname] == colval]


def get_id_codebook(meta_pth: str, synset_pth: str, write_to: str=None):
    "return a dataframe that can be used to translate ILSVRC2012_ID to KERAS_ID"
    columns = ('WNID', 'KERAS_ID', 'ILSVRC2012_ID', 'words')
    dic = dict()

    with open(synset_pth, 'r') as fn:
        for i, line in enumerate(fn):
            s = line.split(' ')[0]
            dic[s] = [i, None, None]

    meta = loadmat(meta_pth)
    names = meta['synsets'][0].dtype.names
    meta_idx = {k: names.index(k) for k in columns if k in names}

    for m in meta['synsets']:
        tmp = dict()
        for key, idx in meta_idx.items():
            tmp[key] = np.squeeze(m[0][idx]).tolist()
        dic.setdefault(tmp['WNID'], [-1, None, None])
        dic[tmp['WNID']][1] = tmp['ILSVRC2012_ID']
        dic[tmp['WNID']][2] = tmp['words']

    df = pd.DataFrame.from_dict(dic, orient='index', columns=columns[1:])
    df = df.sort_values(by=['ILSVRC2012_ID'])
    if write_to is not None:
        df.to_csv(write_to, index_label=columns[0])
    ilsvrc2keras = np.zeros((len(df)+1), dtype='int')
    for i, row in df.iterrows():
        ilsvrc2keras[row['ILSVRC2012_ID']] = row['KERAS_ID']
    return df, ilsvrc2keras


def image2ndarray(fpath: str):
    "read the image file and crop to (224,224,3) numpy array"
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
    return img[:,:,::-1]


class ILSVRC2012ValGen(Sequence):
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
        'Initialization and deal with index translation'
        # load JPEG filenames from folder
        self.list_IDs = [os.path.join(img_folder, x) for x in os.listdir(img_folder)]
        self.list_IDs.sort()
        # deal with index translating
        self.words_table, self.ilsvrc2keras = get_id_codebook(
            os.path.join(devkit_t12, 'data', 'meta.mat'),
            synset_words
            )
        self.ground_truth_table = pd.read_csv(
            os.path.join(devkit_t12, 'data', 'ILSVRC2012_validation_ground_truth.txt'), 
            header=None, names=['ILSVRC2012_ID']
        )
        self.ground_truth_all_kerasid = [self.ilsvrc2keras[i] for i in self.ground_truth_table['ILSVRC2012_ID']]
        # deal with generator parameters
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
        X = np.array([image2ndarray(self.list_IDs[i]) for i in indexes])
        y = np.array([self.ground_truth_all_kerasid[i] for i in indexes])
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_img_words(self, keras_id: int):
        rows = pandas_find_rows(self.words_table, 'KERAS_ID', keras_id)
        assert len(rows) == 1, rows
        return rows['words'][0]