pip install --upgrade pip

cd ./inst_seg/third_party/CenterNet2/
cd ..
python -m pip install -e detectron2

cd ..
pip install -e .

cd ..
pip install -r requirements.txt
pip install -e .
