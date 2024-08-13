pip install --upgrade pip

#pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
#python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html

#pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
#python -m pip install detectron2 -f  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html

pip install -r requirements.txt

git clone https://github.com/keti-ai/ketisdk ../ketisdk
cd ../ketisdk
pip install -e .

