# Create conda environment
```bash
conda create -y -n cvcp python=3.8
conda activate cvcp
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y cuda-toolkit
```
# Download the nuscenes dataset
```bash
# You can use our tar archive or manually wget every link. If manually downloading:
mkdir ./raw & cd ./raw
for i in $(seq -w 01 10); do wget "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval${i}_keyframes.tgz"; done
wget https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz
wget https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval.tgz
wget  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/nuScenes-map-expansion-v1.3.zip"

# Extract dataset
cd .. & mkdir ./nuscenes
for f in $(ls raw/v1.0-*.tgz); do tar -xzvf $f -C ./nuscenes; done
# Install unzip through conda if not already installed
conda -y install unzip

unzip raw/nuScenes-map-expansion-v1.3.zip -d ./nuscenes/maps
```

# Install required modules
```bash
pip install -r requirements.txt
#pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'

cd ./models/centerpoint/ops/iou3d_nms
python setup.py build_ext --inplace
```

# Create metadata (intrinsics, extrinsics, image paths) and labels
```bash
python generate_labels.py
python generate_metadata.py
```

# Ready to train/test!
```bash
python train.py <config_file_path>
python test.py <config_file_path>
```