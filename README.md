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

## HOW TO USE
APP/demo/demo_ketinet.py 참고 
