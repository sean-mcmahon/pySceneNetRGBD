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
import cv2
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

if __name__ == '__main__':

    scenenet_path = '/home/sean/hpc-cyphy/SeanMcMahon/datasets/SceneNet_RGBD/'
    lmbd_name = 'val_rgb_lmdb'
    env = lmdb.open(os.path.join(scenenet_path, lmbd_name), readonly=True)
    img_ls = []
    datum = caffe.proto.caffe_pb2.Datum()
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            print('datakey: ', key)
            if len(img_ls) < 20:
                datum.ParseFromString(value)
                np_img = caffe.io.datum_to_array(datum)
                np_img = np_img.transpose((1, 2, 0))
                np_img = np_img[..., ::-1]
                img_ls.append(np_img)
    env.close()
    print '\n', '-' * 20

    # datum = caffe.proto.caffe_pb2.Datum()
    # datum.ParseFromString(raw_datum)
    #
    # x = np.fromstring(datum.data, dtype=np.uint8)
    print 'shape name ', np.shape(img_ls[0])
    print 'num images', len(img_ls)
    import pdb
    pdb.set_trace()
    cv2.imshow('data', img_ls[5])
    cv2.waitKey(0)
