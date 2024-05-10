docker build -t mmdetection3d .

cd ~/vulcan/driving_preception
sudo docker run --gpus all -it -v ./:/app  -v /dev/shm/wylin2:/mmdetection3d/data/nuscenes mmdetection3d bash

cd /app/config/org
mim download mmdet3d --config [] --dest .

mim download mmdet3d --config centerpoint_pillar02_second_secfpn_head-dcn_8xb4-cyclic-20e_nus-3d --dest .
mim download mmdet3d --config centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d --dest .
mim download mmdet3d --config hv_ssn_regnet-400mf_secfpn_sbn-all_16xb2-2x_nus-3d --dest .
mim download mmdet3d --config hv_ssn_secfpn_sbn-all_16xb2-2x_nus-3d --dest .
mim download mmdet3d --config pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d --dest .


# process dataset
bash preprocess.sh
-- standard
-- g10-v10
-- g10-v20 
...
-- g50-v50



# login to docker
<!-- pip uninstall -y numpy && pip install numpy -->

## build index
cd /mmdetection3d/data/nuscenes/

cp -r standard/samples/ ./
cp -r standard/sweeps/ ./ 
cp -r standard/maps/ ./
if using 'v1.0-mini'
    cp -r standard/v1.0-mini ./
    mv v1.0-mini v1.0-trainval
if using 'v1.0-trainval'
    cp -r standard/v1.0-trainval ./ 

cd /mmdetection3d/
python tools/create_data.py nuscenes --root-path /mmdetection3d/data/nuscenes --out-dir /mmdetection3d/data/nuscenes --extra-tag nuscenes

## run inference
cd /app

python3 /app/test.py  [config] [checkpoint] [data_source]
ex: 

python3 test.py config/centerpoint_circlenms.py config/centerpoint_circlenms.pth --ds g10-v20

#or
bash profile.sh
