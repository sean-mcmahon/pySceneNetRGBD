
import scenenet_pb2 as sn
import numpy as np
import lmdb
import os.path
import pdb


def get_data_proto_paths(data_split='val'):
    split = data_split.split('_', 1)
    data_path = '../{}'.format(split[0])
    pdb.set_trace()
    if len(split) == 2:
        data_path += '/{}'.format(split[1])
    protobuf_path = 'data/scenenet_rgbd_{}.pb'.format(data_split)
    return data_path, protobuf_path


if __name__ == '__main__':
    trajectories = sn.Trajectories()
    dataset = 'val'
    [data_root_path, protobuf_path] = get_data_proto_paths(dataset)
    try:
        with open(protobuf_path, 'rb') as f:
            trajectories.ParseFromString(f.read())
    except IOError:
        print('Scenenet dataset not found at location:{0}'.format(
            data_root_path))
        print('Protobuf file (.pb) not found at: {0}'.format(protobuf_path))
        print('Please ensure you have copied the pb file to the data directory')
