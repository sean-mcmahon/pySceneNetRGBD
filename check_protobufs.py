import scenenet_pb2 as sn
import os
from read_and_shuffle import create_instance_class_maps

if __name__ == '__main__':
    if '/home/sean' in os.path.expanduser('~'):
        cyphy_dir = '/home/sean/hpc-cyphy'
    else:
        cyphy_dir = '/work/cyphy'
    protobuf_path = os.path.join(
        cyphy_dir, 'SeanMcMahon/datasets/SceneNet_RGBD' +
        '/pySceneNetRGBD/data/scenenet_rgbd_val.pb')
    # protobuf_path = '/home/sean/Downloads/scenenet_rgbd_train_10.pb'
    trajectories = sn.Trajectories()
    print 'opening protobuf'
    with open(protobuf_path, 'rb') as f:
        trajectories.ParseFromString(f.read())

    print 'Render Paths:'
    render_paths = [traj.render_path for traj in trajectories.trajectories]
    # print sorted(render_paths)
    print 'Num paths =', len(render_paths)
    matches = [path for path in render_paths if '115' in path]
    print 'paths with "115" in them:\n{}'.format(matches)
    # in train_8, looking for "train/8/8115/instance/100.png"
    mappings = create_instance_class_maps(trajectories)
    if "scenenet_rgbd_train_10" in protobuf_path:
        key = "train/10/10338/instance/100.png"
    else:
        key = "train/8/8115/instance/100.png"
    key_traj = os.path.basename(os.path.abspath(os.path.join(key, os.pardir,
                                                             os.pardir)))
    key_frame_num = os.path.splitext(os.path.basename(key))[0]

    for traj in trajectories.trajectories:
        if len(traj.views) != 300:
            print '{} has {} views'.format(traj.render_path, len(traj.views))
        if key_traj in traj.render_path:
            for view in traj.views:
                if int(key_frame_num) == view.frame_num:
                    print 'key found'
                    break
            break
