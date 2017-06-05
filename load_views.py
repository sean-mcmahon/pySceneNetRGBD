import numpy as np
import lmdb
import os.path
import random
import time
import cv2
import glob
import tarfile
import sys
try:
    import caffe
except ImportError:
    # could be running on hpc
    import subprocess
    cmd = 'module load caffe'
    subprocess.call(cmd, shell=True)
    import caffe
from PIL import Image  # a part of caffe module on HPC
import scenenet_pb2 as sn  # a part of caffe module on HPC


def search_tar(members, pattern):
    """
    Loop over contents find the files with pattern in.
    Return a generator to loop over the img files once.
    """
    print 'searching for', pattern
    matchfound = False
    for tarinfo in members:
        if pattern in tarinfo.name:
            matchfound = True
            print 'match found'
            return
    if not matchfound:
        raise(Exception('Could not find %s in tar file' % pattern))


def tar_type_gen(members, img_type):
    '''
    this gives an iterator for all files of img_type
    '''
    for tarinfo in members:
        if img_type in tarinfo.name and tarinfo.isfile():
            yield tarinfo


def tarobj_to_datum(tarobj, cv_flag=1, compress=True):
    if tarobj is None:
        raise(Exception('Dud (None) tar object given!'))
    byte_arr = bytearray(tarobj.read())
    # cv_flag 1 for load color and cv_flag=2 for load depth
    if cv_flag == 1:
        # decode rgb
        dtype_ = np.uint8
    elif cv_flag == 2:
        # depth or instance
        dtype_ = np.uint16
    else:
        raise(Exception('cv_flag be 1 or 2!'))
    # imdecode gives uint8s and uint16/32s
    cv_img = cv2.imdecode(np.asarray(byte_arr, dtype=dtype_), cv_flag)
    # setting to be float32, I do not think caffe support uint16s and I want all
    # my file types to be the same (all might end up in the same network)
    cv_img = cv_img.astype(np.float32)
    datum = caffe.proto.caffe_pb2.Datum()
    if compress:
        # Need to change image types (to float32), so cannot write compress tar img
        datum.channels = cv_img.shape[2]
        datum.height = cv_img.shape[0]
        datum.width = cv_img.shape[1]
        # cannot reshape and encode, either caffe can handle this or I will write
        # my own data layer.
        datum.float_data.extend(cv2.imencode('.jpg', cv_img)[1])
        datum.encoded = True
    else:
        if cv_img.ndims == 3:
            # rgb image
            im_ = cv_img[:, :, ::-1]  # bgr
            im_ = im_.transpose((2, 0, 1))  # ch, h, w
            datum.channels = im_.shape[0]
        elif cv_img.ndims == 2:
            # depth or label image
            datum.channels = 1
            im_ = cv_img[np.newaxis, ...]
        else:
            raise(Exception('Invalid ndims for cv_img (shape: {})'.format(cv_img.shape)))
        datum.height = im_.shape[1]
        datum.width = im_.shape[2]
        datum.float_data.extend(im_.flattern())
        datum.encoded = False
    return datum


def datum_from_file(img_path, dtype_=None, compress=True):
    if not os.path.isfile(img_path):
        raise(Exception('Invalid File Name: "%s"' % img_path))
    pil_img = Image.open(img_path)
    # do data types have to be the same for caffe input data?
    datum = caffe.proto.caffe_pb2.Datum()
    if dtype_:
        im_ = np.array(pil_img, dtype=dtype_)
    else:
        im_ = np.array(pil_img)

    if compress:
        datum.channels = im_.shape[2]
        datum.height = im_.shape[0]
        datum.width = im_.shape[1]
        datum.data = cv2.imencode('.jpg', im_)
        datum.encoded = True
    else:
        im_ = im_[:, :, ::-1]  # bgr
        im_ = im_.transpose((2, 0, 1))  # ch, h, w
        datum.channels = im_.shape[0]
        datum.height = im_.shape[1]
        datum.width = im_.shape[2]
        datum.data = im_.tobytes()
    return datum


def get_data_proto_paths(data_split='val'):
    split = data_split.split('_', 1)
    data_path = '../{}'.format(split[0])
    if len(split) == 2:
        data_path += '/{}'.format(split[1])
    protobuf_path = 'data/scenenet_rgbd_{}.pb'.format(data_split)
    return data_path, protobuf_path


def loop_over_tar(tar_, lmdb_env, img_type, r_seed=378, early_stop=None):
    img_gen = tar_type_gen(tar_, img_type)
    if img_type.lower() == 'photo':
        flag = 1
    else:
        flag = 2
    rand_lim = 300 * 1000 * 3
    max_num = 1000 * 300
    np.random.seed(r_seed)
    with lmdb_env.begin(write=True) as txn:
        start = time.time()
        for count, img in enumerate(img_gen):
            img_tarobj = tar_.extractfile(img)
            datum = tarobj_to_datum(img_tarobj, cv_flag=flag)
            # str_id = '{}_{}'.format(
            #     np.random.randint(0, rand_lim), img.name)
            str_id = img.name
            print 'Image name: {}\nSize (bytes): {}'.format(img.name, sys.getsizeof(datum.SerializeToString()))
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
            if count % 100 == 0:
                tt = time.time() - start
                print 'Saved {}/~{} images: Took {} s'.format(count, max_num, tt)
                start = time.time()
            if early_stop and count >= early_stop:
                print 'breaking early'
                break
        print '{} images saved.'.format(count)


if __name__ == '__main__':
    if '/home/sean' in os.path.expanduser('~'):
        cyphy_dir = '/home/sean/hpc-cyphy'
    else:
        cyphy_dir = '/work/cyphy'
    sup_dir = os.path.join(cyphy_dir, 'SeanMcMahon/datasets/SceneNet_RGBD')
    data_dir = os.path.join(cyphy_dir, 'SeanMcMahon/datasets/SceneNet_RGBD')
    # data_dir = '/tmp/n8307628'
    trajectories = sn.Trajectories()
    # upper size of rgb ~37,908 bytes
    rgb_max = 37908
    # upper size of depth ~30,585 bytes
    d_max = 30585
    # upper size of label ~3,799 bytes
    label_max = 3799
    img_per_traj = 300
    #  adding cap to make sure nothing crazy is being written (1TB is fine tho)

    dataset = 'val'
    [data_root_path, protobuf_path] = get_data_proto_paths(dataset)
    protobuf_path = os.path.join(sup_dir, 'pySceneNetRGBD', protobuf_path)
    tarfilename = os.path.join(data_dir, dataset + '.tar.gz')
    if not os.path.isfile(protobuf_path):
        raise(Exception('Could not find .pb file @ %s' % protobuf_path))
    if not os.path.isfile(tarfilename):
        raise(Exception('Could not find tar file @ %s' % tarfilename))
    # try:
    #     print 'Loading protobuf'
    #     with open(protobuf_path, 'rb') as f:
    #         trajectories.ParseFromString(f.read())
    # except IOError:
    #     print('Scenenet dataset not found at location:{0}'.format(
    #         data_root_path))
    #     print('Or protobuf file (.pb) not found at: {0}'.format(protobuf_path))
    #     print('Please ensure you have copied the pb file to the data directory')
    #     raise
    num_traj = 1000  # len(trajectories.trajectories)
    # TODO properly determine the required map size
    max_size = 4 * num_traj * img_per_traj * \
        (rgb_max + d_max + label_max)  # ~80GB (quad to make it fit)
    env_rgb = lmdb.open(os.path.join(data_dir, dataset + '_test'),
                        map_size=max_size)
    tar = tarfile.open(tarfilename, 'r')

    overall_time = time.time()
    print 'looping over tar'
    loop_over_tar(tar, env_rgb, 'photo', early_stop=1000)
    print 'saving took {}'.format(time.time() - overall_time)
    env_rgb.close()
    tar.close()
