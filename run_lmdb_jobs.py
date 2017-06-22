import subprocess
import os

if __name__ == '__main__':
    if '/home/sean' in os.path.expanduser('~'):
        cyphy_dir = '/home/sean/hpc-cyphy'
    else:
        cyphy_dir = '/work/cyphy'
    scenenet_path = os.path.join(
        cyphy_dir, 'SeanMcMahon/datasets/SceneNet_RGBD/')

    datasplits = ['train_{}'.format(num) for num in range(17)]
    datasplits.append('val')

    to_rem = ['train_6', 'train_7']
    [datasplits.remove(item) for item in to_rem]
    script_fullname = os.path.join(scenenet_path, 'pySceneNetRGBD',
                                   'shuffle_nyu13_lmdbs.sh')
    if not os.path.isfile(script_fullname):
        raise(Exception('Invalid filepath: %s' % script_fullname))

    id_list = []
    for dataset in datasplits[0:2]:
        job_name = dataset + '_split_sh'
        qsub_call = "qsub -v dataset={} -N {} {}".format(dataset, job_name,
                                                         script_fullname)
        try:
            jobid_ = subprocess.check_output(qsub_call, shell=True)
        except:
            print '****\nError submitting worker job with command \n', qsub_call
            print '****'
            raise
        id_list.append(jobid_)
        print 'Job submitted, Name "{}", Id: {}'.format(
            job_name, jobid_.replace('\n', ''))
        # print '-------- breaking early ---------'
        # break
    print 'qdel command:'
    print 'qdel', " ".join([id_.replace('\n', '') for id_ in id_list])
