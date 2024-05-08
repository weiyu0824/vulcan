docker build -t mmdetection3d .

cd ~/vulcan/driving_preception
sudo docker run --gpus all -it -v ./:/app  -v /dev/shm/wylin2:/mmdetection3d/data/nuscenes mmdetection3d bash

cd /mmdetection3d


mim download mmdet3d --config [] --dest .

mim download mmdet3d --config centerpoint_pillar02_second_secfpn_head-dcn_8xb4-cyclic-20e_nus-3d --dest .
mim download mmdet3d --config centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d --dest .
mim download mmdet3d --config hv_ssn_regnet-400mf_secfpn_sbn-all_16xb2-2x_nus-3d --dest .
mim download mmdet3d --config hv_ssn_secfpn_sbn-all_16xb2-2x_nus-3d --dest .
mim download mmdet3d --config pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d --dest .



# login to docker
pip uninstall -y numpy && pip install numpy

python /mmdetection3d/tools/create_data.py nuscenes --root-path /mmdetection3d/data/nuscenes/standard --out-dir /mmdetection3d/data/nuscenes/standard --extra-tag nuscenes/standard


- 
vim : 
:%s#'/data/nuscenes/,'#data_root,#g
:%s#'/data/nuscenes/'#data_root+'#g

- 0: python /mmdetection3d/demo/pcd_demo.py /mmdetection3d/demo/data/kitti/000008.bin pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth --show


bash /mmdetection3d/tools/dist_test.sh [config] [ckpt] [num_node]
<!-- bash /mmdetection3d/tools/dist_test.sh ./centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py ./centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth 1 -->

1. chance data path in config file 

bash /mmdetection3d/tools/dist_test.sh /app/tmp/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py /app/tmp/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth 1

