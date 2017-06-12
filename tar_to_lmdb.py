import numpy as np
import lmdb
import os.path
import time
import tarfile
import shutil
import argparse
try:
    import caffe
except ImportError:
    # could be running on hpc
    import subprocess
    cmd = 'module load caffe'
    subprocess.call(cmd, shell=True)
    import caffe
from PIL import Image  # a part of caffe module on HPC
import cv2
# import scenenet_pb2 as sn  # a part of caffe module on HPC


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
    cv_img = cv2.imdecode(np.asarray(byte_arr), cv_flag)
    if cv_img.dtype != dtype_:
        e_s = 'Decoded image should have type {}, but is {}'.format(dtype_,
                                                                    cv_img.dtype)
        raise(Exception(e_s))
    if cv_img is None:
        # pil_img = Image.open(io.BytesIO(byte_arr))
        # cv_img = np.asarray(pil_img, dtype=np.uint16)
        # file_img = np.asarray(Image.open('../'+tarobj.name))
        raise(Exception('Image decoding failed.'))

    datum = caffe.proto.caffe_pb2.Datum()
    if compress and cv_flag == 1:
        # Need to change image types (to float32), so cannot write compress tar
        # img
        datum.channels = cv_img.shape[2]
        datum.height = cv_img.shape[0]
        datum.width = cv_img.shape[1]
        # cannot reshape and encode, either caffe can handle this or I will write
        # my own data layer.
        # datum.data = cv2.imencode('.jpg', cv_img)[1].tobytes()
        datum.data = np.asarray(byte_arr).tobytes()
        # bytearray2 = np.fromstring(datum.data, dtype=np.uint8)
        # cv_img2 = cv2.imdecode(bytearray2, cv_flag)
        datum.encoded = True
    else:
        # setting to be float32, I do not think caffe support uint16s
        cv_img = cv_img.astype(np.float)
        if cv_img.ndim == 3:
            # rgb image
            # im_ = cv_img[:, :, ::-1]  # bgr
            # opencv already uses BGR...
            im_ = cv_img.transpose((2, 0, 1))  # ch, h, w
            datum.channels = im_.shape[0]
        elif cv_img.ndim == 2:
            # depth or label image
            datum.channels = 1
            im_ = cv_img[np.newaxis, ...]
        else:
            raise(Exception('Invalid ndims for cv_img (shape: {})'.format(cv_img.shape)))
        datum.height = im_.shape[1]
        datum.width = im_.shape[2]
        datum.float_data.extend(im_.flatten())
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
    max_num = 1000 * 300
    with lmdb_env.begin(write=True) as txn:
        start = time.time()
        for count, img in enumerate(img_gen):
            img_tarobj = tar_.extractfile(img)
            datum = tarobj_to_datum(img_tarobj, cv_flag=flag)
            str_id = img.name
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
            if count % 1000 == 0:
                tt = time.time() - start
                print 'Saved {}/~{} {} images: Took {} s'.format(count, max_num,
                                                                 img_type, tt)
                start = time.time()
            if early_stop and count >= early_stop:
                print 'breaking early'
                break
        print '{} images saved.'.format(count)


def get_img_mem(tarfilename, img_type):
    if img_type != 'photo' and img_type != 'depth' and img_type != 'instance':
        raise(Exception('Unrecoginised img_type (%s), must be ' % img_type +
                        '"photo", "depth" or "instance"'))
    with tarfile.open(tarfilename, 'r|*') as tar_objs:
        # open tar with stream (keeping things quick)
        print 'Getting img filesize...'
        for obj in tar_objs:
            if img_type in obj.name and obj.isfile():
                print 'Found %s img filesize' % img_type
                return obj.size
    raise(Exception('Coud not find file with "%s" in tarobj name' % img_type))


if __name__ == '__main__':
    if '/home/sean' in os.path.expanduser('~'):
        cyphy_dir = '/home/sean/hpc-cyphy'
    else:
        cyphy_dir = '/work/cyphy'
    sup_dir = os.path.join(cyphy_dir, 'SeanMcMahon/datasets/SceneNet_RGBD')
    data_dir = '/tmp/n8307628'
    if not os.path.isdir(data_dir):
        data_dir = os.path.join(
            cyphy_dir, 'SeanMcMahon/datasets/SceneNet_RGBD')
    parser = argparse.ArgumentParser()
    def_tar = os.path.join(sup_dir, 'val.tar.gz')
    parser.add_argument('--tarname', default=def_tar)
    parser.add_argument('--lmdb_dir', default=sup_dir)
    args = parser.parse_args()
    if not os.path.isdir(args.lmdb_dir):
        raise(Exception('Invalid LMDB path: "{}"'.format(args.lmdb_dir)))
    dataset = os.path.splitext(
        os.path.splitext(os.path.basename(args.tarname))[0])[0]
    tarfilename = args.tarname
    if not os.path.isfile(tarfilename):
        raise(Exception('Could not find tar file @ %s' % tarfilename))

    num_traj = 1000
    img_per_traj = 300
    time_dict = {}
    for im_type in ('photo', 'instance', 'depth'):
        lmdb_path = os.path.join(args.lmdb_dir, dataset + '_' + im_type + '_lmdb')
        print '\n\n', '=' * 50
        print 'Saving {} images to LMDB {}\n'.format(im_type,
                                                     os.path.basename(lmdb_path))
        img_mem_ = get_img_mem(tarfilename, im_type)
        # sometimes save imgs as float32, 4 times more memory than uint8 RGB pixels)
        # and double for breathing room
        max_size = 2 * num_traj * img_per_traj * 4 * img_mem_
        if max_size < 1.1565e+11:
            print 'max size of "{}" may not be enough, overwriting'.format(max_size)
            max_size = 1.1565e+11
        if os.path.isdir(lmdb_path):
            shutil.rmtree(lmdb_path)
        with lmdb.open(lmdb_path, map_size=max_size) as lmbd_env, tarfile.open(tarfilename, 'r') as tar:
            overall_time = time.time()
            print 'looping over tar'
            loop_over_tar(tar, lmbd_env, im_type)
            time_dict[im_type + '_time'] = time.time() - overall_time
            print 'saving took {}'.format(time_dict[im_type + '_time'])
    for key, item in time_dict.iteritems():
        print '{} = {}s'.format(key, item)
