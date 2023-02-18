conda create -n centernet2 python=3.7
conda activate centernet2
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
python -m pip install detectron2 -f \
https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html
