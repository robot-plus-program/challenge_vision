![license](https://img.shields.io/badge/license-MIT-green) ![PyTorch-1.4.0](https://img.shields.io/badge/PyTorch-1.4.0-blue)
# KPICK -  AI-BASED PICKING PACKAGE

## System Requirements and prerequiste
```sh
- CUDA >=11.0, CUDNN>=7
```

## Installation
### KETINET
pytorch (pytorch >= 1.4.0, pytorch version ref. https://pytorch.org/get-started/previous-versions/ )
```commandline
pip install $pytorch_version$
```
```commandline
pip install Cython

cd kpick-devel
pip install -e .

cd ../ketisdk
pip install -e .
```

### Detic
```commandline
cd inst_seg
sh install.sh
```
download https://drive.google.com/file/d/1SkQLeo3CelUUNpxpNZxZ94Jq3mewDGD6/view?usp=drive_link
```angular2html
mv ./Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth ./inst_seg/inst_seg/models
```
## HOW TO USE
```angular2html
APP/demo/demo_ketinet.py 참고 
APP/demo/demo_detic.py 참고 
```
