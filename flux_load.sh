module load python-dev/3.5.2
module load cuda/7.5

export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc0-cp35-cp35m-linux_x86_64.whl
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/sw/arcts/centos7/modulefiles/cuda/lib64"

pip install --user --upgrade $TF_BINARY_URL
