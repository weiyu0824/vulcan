from op import Loader, Detector
import socket

def get_detector_args():
    return {
        'model_name': 'yolov8n'
    }

def get_loader_args():
    return {
        'batch_size': 8
    }


def main():
    num_profiled_batch = 3

    loader = Loader(get_loader_args())
    detector = Detector(get_detector_args())

    for batch_idx in range(num_profiled_batch):
        # print(batch_idx)s
        batch_data = loader.load_batch()
        batch_data = detector.profile(batch_data) 

    # 
    profile_dict = {}
    
def start_tcp_server(host, port):
    # Create a TCP/IP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # Bind the socket to the address and port
        server_socket.bind((host, port))

        # Listen for incoming connections
        server_socket.listen()

        print(f"Server listening on {host}:{port}")

        # Accept a new connection
        client_socket, client_address = server_socket.accept()

        print(f"Connection from {client_address}")

        # Handle the connection (you can implement your own logic here)
        # For example, you can read/write data from/to the client_socket
        # client_socket.recv() to receive data
        # client_socket.send() to send data
        cmd = client_socket.recv(1024)
        print(cmd)


        client_socket.send({
            'cpu_usage': 10
        })

        # Close the client socket
        client_socket.close() 


    # img_path = './nuimages/samples/CAM_BACK_LEFT/n008-2018-05-30-16-31-36-0400__CAM_BACK_LEFT__1527712679297295.jpg'
    

    # # img_source = np.array(Image.open(img_path))
    # img_source = cv2.imread(img_path)
    # img_source = cv2.resize(img_source, (640, 640))
    # img_source = img_source.astype(np.float32) / 255.0

    # print('img source', img_source.shape)
    # batch_imgs = []
    # # batch_imgs.append(img_source.tolist())
    # for i in range(256):
    #     batch_imgs.append(img_source)
    # batch_tensor = np.array(batch_imgs)
    # batch_tensor = np.transpose(batch_tensor, (0, 3, 1, 2))

    # print('batch tensor', batch_tensor.shape)

    # batch_data = {
    #     'images': batch_tensor,
    #     'labels': []
    # }
    # import time
    # st = time.time()
    # for i in range(10):
    #     detector.profile(batch_data)
    # print(time.time() - st)

if __name__ == "__main__":
    addr, port = 'localhost', 12345
    start_tcp_server(addr, port)