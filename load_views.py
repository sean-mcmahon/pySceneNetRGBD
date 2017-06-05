
import scenenet_pb2 as sn
import numpy as np
import lmdb
import os.path
import random
import time
import cv2
import glob
import tarfile
try:
    import caffe
except ImportError:
    # could be running on hpc
    import subprocess
    cmd = 'module load caffe'
    subprocess.call(cmd, shell=True)
    import caffe
from PIL import Image  # a part of caffe module on HPC


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


def tarobj_to_datum(tarobj, cv_flag=1):
    if tarobj is None:
        raise(Exception('Dud (None) tar object given!'))
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


def get_datum(img_path, dtype_=None):
    if not os.path.isfile(img_path):
        raise(Exception('Invalid File Name: "%s"' % img_path))
    pil_img = Image.open(img_path)
    # do data types have to be the same for caffe input data?
    if dtype_:
        im_ = np.array(pil_img, dtype=dtype_)
    else:
        im_ = np.array(pil_img)
    im_ = im_[:, :, ::-1]  # bgr
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
        for count, img in enumerate(img_gen):
            img_tarobj = tar_.extractfile(img)
            datum = tarobj_to_datum(img_tarobj, cv_flag=flag)
            # str_id = '{}_{}'.format(
            #     np.random.randint(0, rand_lim), img.name)
            str_id = img.name
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
            if count % 100 == 0:
                print 'Saved {}/~{} images'.format(count, max_num)
            if early_stop and count >= early_stop:
                print 'breaking early'
                break


if __name__ == '__main__':
    if '/home/sean' in os.path.expanduser('~'):
        cyphy_dir = '/home/sean/hpc-cyphy'
    else:
        cyphy_dir = '/work/cyphy'
    data_dir = os.path.join(cyphy_dir, 'SeanMcMahon/datasets/SceneNet_RGBD')
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
    protobuf_path = os.path.join(data_dir, 'pySceneNetRGBD', protobuf_path)
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
    max_size = 2 * num_traj * img_per_traj * \
        (rgb_max + d_max + label_max)  # ~6GB (double for safety)
    env_rgb = lmdb.open(os.path.join(data_dir, dataset + '_tar_rgb_lmdb'),
                        map_size=max_size)
    tar = tarfile.open(tarfilename, 'r')

    # nested for loop. Outer loop over random frame_ids inner loops over shuffles
    # trajectories iterable.
    r_frame_ids = random.sample(xrange(img_per_traj), img_per_traj)
    # random.shuffle(trajectories.trajectories)  # error AttError setitem
    # rand_trajectories = random.sample(trajectories.trajectories, num_traj)
    overall_time = time.time()
    # for f_count, frame_id in enumerate(r_frame_ids):
    #     txn = env_rgb.begin(write=True)
    #     traj_loop_t = time.time()
    #     print 'Saving frame_id {} into trajectories. {}/{} ids\n'.format(
    #         frame_id * 25, f_count, len(r_frame_ids))
    #     save_time = time.time()
    #     for count, traj in enumerate(rand_trajectories):
    #         view = traj.views[frame_id]
    #         img_name = str(os.path.join(data_dir, dataset, traj.render_path,
    #                                     'photo', str(view.frame_num)))
    #         img_name = glob.glob(img_name + '*')[0]
    #         datum = get_datum(img_name, dtype_=np.uint8)
    #         str_id = img_name.replace('/', '-')
    #         try:
    #             txn.put(str_id.encode('ascii'), datum.SerializeToString())
    #         except:
    #             print 'Error saving image: %s\nstr id: %s\n' % img_name, str_id
    #             raise
    #         if count % 50 == 0:
    #             time_ = time.time() - save_time
    #             print 'Processed {}/{} images. Time: {} seconds'.format(count, num_traj, time_)
    #             save_time = time.time()
    #     txn.commit()
    #     # print 'Avg extract time: {}\nAvg datum process time: {}'.format(
    #     #     sum(tartimes) / len(tartimes), sum(dattimes) / len(dattimes))
    #     print '\nFrame completed time: {} seconds '.format(time.time() - traj_loop_t)
    #     print '-' * 50
    #     if f_count >= 2:
    #         print 'breaking early'
    #         break
    print 'looping over tar'
    loop_over_tar(tar, env_rgb, 'photo')
    print 'saving took {}'.format(time.time() - overall_time)
    env_rgb.close()
    tar.close()
