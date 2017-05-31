
import scenenet_pb2 as sn
import numpy as np
import lmdb
import os.path
import pdb
import random
import tarfile
import io
try:
    import caffe
except ImportError:
    # could be running on hpc
    import subprocess
    cmd = 'module load caffe'
    subprocess.call(cmd, shell=True)
    import caffe
from PIL import Image  # a part of caffe module on HPC


def get_data_proto_paths(data_split='val'):
    split = data_split.split('_', 1)
    data_path = '../{}'.format(split[0])
    if len(split) == 2:
        data_path += '/{}'.format(split[1])
    protobuf_path = 'data/scenenet_rgbd_{}.pb'.format(data_split)
    return data_path, protobuf_path


if __name__ == '__main__':
    error_check = True
    trajectories = sn.Trajectories()
    # upper size of rgb ~37,908 bytes
    rgb_max = 37908
    # upper size of depth ~30,585 bytes
    d_max = 30585
    # upper size of label ~3,799 bytes
    label_max = 3799
    num_scenes = 153
    img_per_traj = 300
    #  adding cap to make sure nothing crazy is being written (1TB is fine tho)
    max_size = 2 * num_scenes * img_per_traj * \
        (rgb_max + d_max + label_max)  # ~6GB
    env = lmdb.open('train_0', map_size=max_size)

    dataset = 'train_0'
    [data_root_path, protobuf_path] = get_data_proto_paths(dataset)
    try:
        with open(protobuf_path, 'rb') as f:
            trajectories.ParseFromString(f.read())
    except IOError:
        print('Scenenet dataset not found at location:{0}'.format(
            data_root_path))
        print('Protobuf file (.pb) not found at: {0}'.format(protobuf_path))
        print('Please ensure you have copied the pb file to the data directory')

    # get random idx's into train set
    num_imgs = len(trajectories.trajectories) * \
        img_per_traj  # 300 imgs per traj
    print 'generating {} random (non repeating) indices'.format(num_imgs)
    view_idx = random.sample(xrange(img_per_traj), img_per_traj)
    random.shuffle(trajectories.trajectories)
    tar = tarfile.open(tarfilename, 'r')
    for vi in view_idx:
        txn = env.begin(write=True)
        for traj in trajectories.trajectories:
            view = traj.views[vi]
            for img_type, n_c in zip(['photo', 'depth', 'instance'],
                                     [3, 1, 1]):
                tar_img_name = os.path.join(traj.render_path, img_type,
                                            view.frame_num)
                img = tar.getmember(tar_img_name)
                byte_arr = bytearray(img.read())
                if error_check:
                    pil_img = Image.open(io.BytesIO(byte_arr))
                    np_img = np.array(pil_img)
                    if n_c != np_img.shape(2):
                        print 'img shape: ', np.shape(np_img)
                        raise(
                            Exception(
                                'expected num of channels does not match'))
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = n_c
                datum.height = 240
                datum.width = 320
                # TODO caffe takes weird shapes of arrays! Check if this
                # works
                datum.data = byte_arr
                datum.encoded = True
                str_id = tar_img_name.replace('/', '-')
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
                print 'Breaking img_type loop Early!'
                break
            print 'Breaking traj loop Early!'
            break
        txn.close()
        print 'Breaking view loop Early!'
        break
    tar.close()
