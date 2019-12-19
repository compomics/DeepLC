wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ./Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p ./Miniconda3
Miniconda3/condabin/conda create -n deeplc_gui python=3.7
Miniconda3/envs/deeplc_gui/bin/pip install deeplc

