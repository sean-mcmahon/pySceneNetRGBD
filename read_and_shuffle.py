"""
This script will read unshuffed images from LMDBs, shuffle, do some processing
and then save to new LMDBs.
Key Tasks:
- Read in images from the three LMDBs for each tar
- Append random int to keys to shuffle, same ints to same keys accross img types
- Convert the Instance images to NYU 13 labels
- Make sure Image data types compabible with Caffe (float and uint8 for label)
- Check status of images, compare to img read from file (val only)

By Sean McMahon, 7th June 2017
"""
import numpy as np
import lmdb
import os
try:
    import caffe
except ImportError:
    # could be running on hpc
    import subprocess
    cmd = 'module load caffe'
    subprocess.call(cmd, shell=True)
    import caffe
from PIL import Image  # a part of caffe module on HPC
import scenenet_pb2 as sn
import cv2


def get_img(raw_string):
    datum.ParseFromString(raw_string)
    np_img = np.fromstring(datum.data, dtype=np.uint8).reshape(
        datum.channels, datum.height, datum.width)(datum)
    if np_img.ndim == 3:
        # colour
        np_img = np_img.transpose((1, 2, 0))
        np_img = np_img[..., ::-1]
    elif np.img.ndim == 2:
        # depth or greyscale
        raise(Exception('Code incomplete.'))

    return np_img


def checkImg(key, value):
    """
    Compares LMDB image with image read from file. Making sure they are the same.
    Can only process for the validation set.
    Check image format as well, should be float or uint8
    """
    return False


def convert_nyu(key, value, trajectories):
    """
    Read label/instance image and convert from wordnet labels to NYU13.
    Based on convert_instance2class.py
    """
    pass

def makefloat():
    """
    If the image is not a float convert
    """
    pass

if __name__ == '__main__':

    scenenet_path = '/home/sean/hpc-cyphy/SeanMcMahon/datasets/SceneNet_RGBD/'
    lmdb_path = '/home/sean/hpc-cyphy/SeanMcMahon/datasets/SceneNet_RGBD/'
    lmdb_names = ['val_instance_lmdb', 'val_rgb_lmdb', 'val_depth_lmdb']
    img_LMDBs = [os.path.join(lmdb_path, lmdb_name)
                 for lmdb_name in lmdb_names]
    for lmdb_name in img_LMDBs:
        env = lmdb.open(lmdb_name, readonly=True)
        [path, tail] = os.path.split(lmdb)
        new_name = path + 'shuffle_' + tail
        new_env = env = lmdb.open(new_name)

        np.random.seed(9421)
        with env.begin() as txn, new_env.begin(write=True) as w_txn:
            cursor = txn.cursor()
            # move to start of database
            if not cursor.first():
                raise(Exception('Could locate beginning of database, could be empty'))
            for key, value in cursor:
                # check and format images
                if not checkImg(key, value):
                    print 'Issue with: ', key, ' in ', lmdb_name
                    raise(Exception('Image in LMDB different to image in file'))
                if 'instance' in key.lower():
                    n_value = convert_nyu(key, value, trajectories)
                else:
                    n_value = makefloat(value)
                r_int = np.random.randint(0, num_imgs)
                n_key = '{0:06d}_{}'.format(r_int, key)
                # write to new lmdb
                w_txn.put(n_key, n_value)
