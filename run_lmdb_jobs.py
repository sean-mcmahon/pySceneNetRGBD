import numpy as np
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
    
    # these lmdbs do not exist or are incomplete
    to_rem = ['val', 'train_6', 'train_12', 'train_0', 'train_1', 'train_2']
    for item in to_rem:
        datasplits.remove(item)
    script_fullname = os.path.join(scenenet_path, 'pySceneNetRGBD',
                                   'shuffled_nyu13_lmdbs.sh')

    for dataset in datasplits:
        job_name = 'sh_' + dataset
        qsub_call = "qsub -v dataset={} -N {} {}".format(dataset, job_name,
                                                         script_fullname)
        try:
            jobid_ = subprocess.check_output(qsub_call, shell=True)
        except:
            print '****\nError submitting worker job with command \n', qsub_call
            print '****'
            raise
        print 'Job submitted, Id: {}'.format(jobid_)
