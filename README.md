# Vulcan Implmentation

# Prepare environments: 
1. GPU driver:
    ```
    bash sudo script/install_nvdriver.sh
    ```
    ```
    sudo reboot
    ```
    - check if driver correctly installed by command `nvidia-smi`
2. Python: 
    ```
    sudo bash script/install_pip.sh
    ```
3. Docker (optional)
    ```
    sudo bash script/install_docker_gpu.sh
    sudo bash script/enable_docker_rootless.sh
    ```
4. Environments: 
    - conda
        ```
        conda create -n vulcan python=3.10
        conda create -f vulcan.yaml # or from file, modified the prefix field if needed
        ```

# Searching alogirthm
## Single query search
Both search_by_utility & search_by_accuracy are functions used to search best query_setup for single query. 
- search_by_utility: metric=utility defined in vulcan
- search_by_accuracy: self-defined metric excluded latency
```
python3 multi-engine.py
```
```python3
# sample code to use single query search.
# see multi_engine.py to have better understanding.

engine = Engine(queries=[], cluster_spec=cluster_spec, num_profile_sample=5000) 
query_setups = engine.search_by_utlity(queries[0], ClusterState(cluster_spec))
query_setups = sorted(query_setups, key = lambda x: x.setup_metric.utility, reverse=True);
print(query_setups[0])
```
