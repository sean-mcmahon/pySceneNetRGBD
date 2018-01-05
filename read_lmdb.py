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

if __name__ == '__main__':
    scenenet_path = '/home/sean/hpc-cyphy/SeanMcMahon/datasets/SceneNet_RGBD/'
    lmdb_names = ['val_1000_instance',
                  'val_1000_depth', 'val_1000_photo']
    img_LMDBs = [os.path.join(scenenet_path, lmdb_name)
                 for lmdb_name in lmdb_names]
    img_LMDBs = '/work/cyphy/SeanMcMahon/datasets/SceneNet_RGBD/shuffled_nyu13_lmdbs/train_8_instance_lmdb_shuffled_NYU13'
    for lmbd_name in img_LMDBs:
        print '\nLoading {}'.format(os.path.basename(lmbd_name))
        env = lmdb.open(lmbd_name, readonly=True)
        img_ls = []
        datum = caffe.proto.caffe_pb2.Datum()
        with env.begin() as txn:
            cursor = txn.cursor()
            for count, (key, value) in enumerate(cursor):

                # if len(img_ls) < 20:
                #     datum.ParseFromString(value)
                #     if datum.encoded:
                #         pass
                #     else:
                #         np_img = caffe.io.datum_to_array(datum)
                #         np_img = np_img.transpose((1, 2, 0))
                #         np_img = np_img[..., ::-1]
                #         img_ls.append(np_img)
                if count % 200 == 0:
                    print('datakey: ', key)
                    count_b = count
        env.close()
        print 'count = ', count_b
        print '\n', '-' * 20

    # datum = caffe.proto.caffe_pb2.Datum()
    # datum.ParseFromString(raw_datum)
    #
    # x = np.fromstring(datum.data, dtype=np.uint8)
