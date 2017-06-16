'''
This script will check all views and traj in protobuf files.
'''
import scenenet_pb2 as sn
import os
import glob

if __name__ == '__main__':
    if '/home/sean' in os.path.expanduser('~'):
        cyphy_dir = '/home/sean/hpc-cyphy'
    else:
        cyphy_dir = '/work/cyphy'

    search_path = os.path.join(
        cyphy_dir, 'SeanMcMahon/datasets/SceneNet_RGBD' +
        '/pySceneNetRGBD/data/')
    protobuf_paths = glob.glob(search_path + '*.pb')
    # protobuf_paths = [os.path.join(search_path,'scenenet_rgbd_train_16.pd')]
    # protobuf_paths = ['/home/sean/Downloads/train_protobufs/scenenet_rgbd_train_16.pb']
    # protobuf_paths = [os.path.join(
    #     cyphy_dir, 'SeanMcMahon/datasets/SceneNet_RGBD' +
    #     '/pySceneNetRGBD/data/scenenet_rgbd_train_8.pb')]
    # protobuf_paths.append(os.path.join(
    #     cyphy_dir, 'SeanMcMahon/datasets/SceneNet_RGBD' +
    #     '/pySceneNetRGBD/data/scenenet_rgbd_train_10.pb'))
    # protobuf_paths.append(os.path.join(
    #     cyphy_dir, 'SeanMcMahon/datasets/SceneNet_RGBD' +
    #     '/pySceneNetRGBD/data/scenenet_rgbd_val.pb'))

    trajectories = sn.Trajectories()
    if len(protobuf_paths) != 18:
        print 'incorrect number of protobuf_paths found:'
        for p in protobuf_paths:
            print(p)
        print 'Warning: {} files found'.format(len(protobuf_paths))
    protobuf_paths = [protobuf_paths[-1]]
    dud_traj = []
    for proto in protobuf_paths:
        dud = False
        trajectories = sn.Trajectories()
        print 'opening protobuf: "{}"...'.format(proto)
        with open(proto, 'rb') as f:
            trajectories.ParseFromString(f.read())

        if len(trajectories.trajectories) != 1000:
            print '{} has {} trajectories'.format(os.path.basename(proto), len(trajectories.trajectories))
            dud = True

        for traj in trajectories.trajectories:
            if len(traj.views) != 300:
                print '{} has {} views'.format(traj.render_path, len(traj.views))
                dud = True

        if dud:
            dud_traj.append(proto)
        else:
            print 'no issues found with: {}'.format(os.path.basename(proto))

    print '{} duds found:'.format(len(dud_traj))
    for traj in sorted(dud_traj):
        print traj
