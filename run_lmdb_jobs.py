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
