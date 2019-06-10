# load modules
module load gcc/4.8.5 python_gpu/3.6.4 hdf5 eth_proxy
module load cudnn/7.2

# install pacakges
pip install --user virtualenvwrapper
source $HOME/.local/bin/virtualenvwrapper.sh
mkvirtualenv "nlu-project2"

export PATH=$PATH:/$PWD
