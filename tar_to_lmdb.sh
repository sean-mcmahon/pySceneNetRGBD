#!/bin/bash -l
#PBS -N val_proc
#PBS -l ncpus=1
#PBS -l mem=64GB
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
else
  echo "No directory found..."
fi
# declare -a arr=("3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16")
# for ii in "${arr[@]}"
# do
#   tar_name='train_'$ii'.tar.gz'
#   echo $tar_name
#    # or do whatever with individual element of the array
# done

tar_name='val.tar.gz'
tmp_dir='/tmp/n8307628/'
#
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
