# Vulcan Implmentation

## Prepare environments: 
1. GPU driver:
    ```
    bash sudo script/install_nvdriver.sh
    ```
    ```
    sudo reboot
    ```
    - check by nvidia-smi
2. Python: 
    ```
    sudo bash script/install_pip.sh
    ```
3. Docker (optional)
    ```
    sudo bash script/install_docker_gpu.sh
    sudo bash script/enable_docker_rootless.sh
    ```