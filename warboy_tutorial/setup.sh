sudo apt-get update && sudo apt-get install cmake libeigen3-dev libgl1-mesa-glx

wget -O part5/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt

pip install -r requirements.txt

cd part5/utils/postprocess_func/cbox_decode
rm -rf build cbox_decode.so
python build.py build_ext --inplace
cd -


git clone https://github.com/furiosa-ai/warboy-vision-models.git ./part5/warboy-vision-models
cd part5/warboy-vision-models
git switch furiosa_demo
./build.sh
cd -

mv demo.yaml part5/warboy-vision-models/cfg/demo.yaml

