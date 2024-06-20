# Nuimage Object Detection
## Dataset Preperation
1. Download datasets
    - nuimages-v1.0-all-metadata.tgz  
    - nuimages-v1.0-all-samples.tgz 
    - Extract: ```tar -xzf <filename>```
2. Download requirements
    - ```pip install -r requirements.txt```

## Run without Container
- step1: build index
    ```
    python3 main.py -m build
    ```
    then you will get a file call `v1.0-<version>-idx.json`
- step2: profile
    ```
    python3 main.py -m profile
    ```
Note: modify the data_root & version at op.py


## Run Container
sudo docker build -t video-task . 

sudo docker run -v /dev/shm/wylin/nuimages:/nuimages video-task
```
sudo docker run -it -v /dev/shm/wylin/nuimages:/nuimages -e DETECTION_MODEL="yolov8n" video-task /bin/bash
```

sudo docker run --gpus all -v /dev/shm/wylin/nuimages:/nuimages -v /users/wylin2/vulcan/video_analytics:/app --network host  -it video-task /bin/bash

priviledge: 
bash ~/scripts/enable_rootless.sh 
