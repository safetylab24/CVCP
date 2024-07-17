# Create conda environment
conda create -n CVCP3 python=3.8
conda activate CVCP3
conda install pytorch torchvision pytorch-cuda=12.1 cuda-toolkit=12.1 -c pytorch -c nvidia

# Install required modules
pip install -r requirements.txt
<!-- pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable' -->

# Build cuda util from source
cd ./models/centerpoint/ops/iou3d_nms
python setup.py build_ext --inplace

# Ready to train/test!
python train.py <config_file_path>
python test.py <config_file_path>