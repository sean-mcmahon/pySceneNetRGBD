#!/bin/bash -l
#PBS -N train_5_proc
#PBS -l ncpus=1
#PBS -l mem=32GB
#PBS -l walltime=48:00:00
module load python/2.7.11-foss-2016a
module load caffe

wget=/usr/bin/wget

cyphy_dir='/work/cyphy'
local_dir='/home/sean'
if [[ -d $local_dir ]]; then
  working_dir=$local_dir'/hpc-cyphy/SeanMcMahon/datasets/SceneNet_RGBD/'
  python_script=$working_dir'pySceneNetRGBD/tar_to_lmdb.py'
elif [[ -d $cyphy_dir ]]; then
  working_dir=$cyphy_dir'/SeanMcMahon/datasets/SceneNet_RGBD/'
  python_script=$working_dir'pySceneNetRGBD/tar_to_lmdb.py'
  # Because using MKL Blas on HPC
  export MKL_CBWR=AUTO
else
  echo "No directory found..."
fi


tar_name='train_5.tar.gz'
tmp_dir='/tmp/n8307628/'

if [[ -e $working_dir$tar_name ]];
then
  echo 'Tar file exsists'
else
  echo "filename:"$working_dir$tar_name
  $wget 'http://www.doc.ic.ac.uk/~ahanda/train_split/'$tar_name --directory-prefix=$working_dir
fi

mkdir -p $tmp_dir
cp $working_dir$tar_name $tmp_dir -v

python $python_script --tarname $tmp_dir$tar_name --lmdb_dir $working_dir
