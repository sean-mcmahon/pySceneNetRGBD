#!/bin/bash -l
#PBS -l ncpus=1
#PBS -l mem=64GB
#PBS -l walltime=48:00:00
module load python/2.7.11-foss-2016a
module load caffe

cyphy_dir='/work/cyphy'
local_dir='/home/sean'
if [[ -d $local_dir ]]; then
  working_dir=$local_dir'/hpc-cyphy/SeanMcMahon/datasets/SceneNet_RGBD/'
  python_script=$working_dir'pySceneNetRGBD/read_and_shuffle.py'
elif [[ -d $cyphy_dir ]]; then
  working_dir=$cyphy_dir'/SeanMcMahon/datasets/SceneNet_RGBD/'
  python_script=$working_dir'pySceneNetRGBD/read_and_shuffle.py'
else
  echo "No directory found..."
fi

# dataset="$1"
if [[ -z "$dataset" ]]; then
  echo 'Must give a dataset'
  exit 1
fi
# lmdb_path="$2"
if [[ -z "$lmdb_path" ]]; then
  lmdb_path=$working_dir
fi
# out_dir="$3"
if [[ -z "$out_dir" ]]; then
  out_dir=$working_dir'shuffled_nyu13_lmdbs/'
fi
echo 'dataset='$dataset
echo 'lmdb_path='$lmdb_path
echo 'out_dir='$out_dir

mkdir -p $out_dir
python $python_script --in_lmdb_dataset $dataset --in_lmdb_path $lmdb_path --lmdb_out_path $out_dir
