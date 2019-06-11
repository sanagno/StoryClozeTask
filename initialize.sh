# load modules
module load gcc/4.8.5 python_gpu/3.6.4 hdf5 eth_proxy
module load cudnn/7.2

# pip install --user virtualenvwrapper
# source $HOME/.local/bin/virtualenvwrapper.sh
# mkvirtualenv "nlu-project2"

# install pacakges
python -m venv "nlu-project2"
source nlu-project2/bin/activate

export PATH=$PATH:/$PWD
