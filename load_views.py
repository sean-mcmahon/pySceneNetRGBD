
import scenenet_pb2 as sn
import numpy as np
import lmdb
import os.path
import random
import tarfile
import time
import cv2
try:
    import caffe
except ImportError:
    # could be running on hpc
    import subprocess
    cmd = 'module load caffe'
    subprocess.call(cmd, shell=True)
    import caffe
from PIL import Image  # a part of caffe module on HPC


def tarobj_to_datum(tarobj, cv_flag=1):
    byte_arr = bytearray(tarobj.read())
    # cv_flag 1 for load color and cv_flag=2 for load depth
    if cv_flag == 1:
        # colour image
        dtype_ = np.uint8
    elif cv_flag == 2:
        # depth or instance image
        dtype_ = np.uint16  # depth images are uin16s (cv takes 16 or 32 bit)
    else:
        raise(Exception('cv_flag be 1 or 2!'))
    cv_img = cv2.imdecode(np.asarray(byte_arr, dtype=dtype_), cv_flag)
    im_ = cv_img[:, :, ::-1]  # bgr
    im_ = im_.transpose((2, 0, 1))  # ch, h, w
    datum = caffe.proto.caffe_pb2.Datum()
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


if __name__ == '__main__':
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
    tarfilename = '../{}.tar.gz'.format(dataset)
    if not os.path.isfile(tarfilename):
        raise(Exception('Invalid tarfilename: %s' % tarfilename))
    try:
        with open(protobuf_path, 'rb') as f:
            trajectories.ParseFromString(f.read())
    except IOError:
        print('Scenenet dataset not found at location:{0}'.format(
            data_root_path))
        print('Or protobuf file (.pb) not found at: {0}'.format(protobuf_path))
        print('Please ensure you have copied the pb file to the data directory')
    num_traj = len(trajectories.trajectories)
    max_size = 2 * num_traj * img_per_traj * \
        (rgb_max + d_max + label_max)  # ~6GB (double for safety)
    env_rgb = lmdb.open(dataset + '_lmdb', map_size=max_size)

    # nested for loop. Outer loop over random frame_ids inner loops over shuffles
    # trajectories iterable.
    r_frame_ids = random.sample(xrange(img_per_traj), img_per_traj)
    # random.shuffle(trajectories.trajectories)  # error AttError setitem
    rand_trajectories = random.sample(trajectories.trajectories, num_traj)
    tar = tarfile.open(tarfilename, 'r')
    for f_count, frame_id in enumerate(r_frame_ids):
        txn = env_rgb.begin(write=True)
        print 'Saving frame_id {} into trajectories. {}/{}'.format(
            frame_id * 25, f_count, len(r_frame_ids))
        tartimes = []
        dattimes = []
        for count, traj in enumerate(rand_trajectories):
            view = traj.views[frame_id]
            tar_img_name = str(os.path.join(traj.render_path, 'photo',
                                            str(view.frame_num)))
            tartime = time.time()
            tarobj = tar.extractfile(tar_img_name)
            tartimes.apend(time.time() - tartime)
            dattime = time.time()
            datum = tarobj_to_datum(tarobj, cv_flag=1)
            dattimes.apend(time.time() - dattime)
            str_id = tar_img_name.replace('/', '-')
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
            # if count % 10 == 0:
            print 'Processed {}/{} images'.format(count + 1, num_traj)
        txn.commit()
        print 'Avg extract time: {}\nAvg datum process time: {}'.format(
            sum(tartimes) / len(tartimes), sum(dattimes)/len(dattimes))
        print '-' * 50
    tar.close()
    env_rgb.close()
