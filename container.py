import docker

def check_container_status(remote_host, container_id):
    client = docker.DockerClient(base_url=f"tcp://{remote_host}:2375")
    try:
        container = client.containers.get(container_id)
        if container.status == "running":
            print("Container is running.")
        else:
            print("Container is not running.")
    except container.errors.NotFound:
        print("Container not found.")
    except container.errors.APIError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    remote_host = "c240g5-110201.wisc.cloudlab.us"  # IP address of the remote host
    container_id = "66f55d54284d"    # ID of the Docker container

    check_container_status(remote_host, container_id)