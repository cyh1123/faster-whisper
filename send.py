import socket
import threading

# 服务器的主机地址和端口号
HOST = 'localhost'  # 或者服务器的IP地址
PORT = 43007         # 服务器监听的端口，根据实际情况进行替换

# 分块大小，每次发送的字节数
CHUNK_SIZE = 65536

def send_audio_data(sock, file_path):
    """发送音频数据到服务器."""
    with open(file_path, 'rb') as audio_file:
        while True:
            # 读取音频文件的一块数据
            audio_chunk = audio_file.read(CHUNK_SIZE)
            if not audio_chunk:
                break  # 文件读取完毕
            sock.sendall(audio_chunk)
    # 可以在这里发送一个特殊信号给服务器，表示发送完毕
    # 发送结束信号
    sock.sendall(b'END')

def receive_server_response(sock):
    """接收服务器的响应数据."""
    while True:
        response_data = sock.recv(CHUNK_SIZE)
        if not response_data:
            break  # 服务器关闭连接或发送完成
        print('Received:', response_data.decode())
    # 可以在这里处理服务器发送的完整响应

# 创建一个socket连接
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    # 连接到服务器
    client_socket.connect((HOST, PORT))
    
    # 创建一个线程来接收服务器的响应
    receive_thread = threading.Thread(target=receive_server_response, args=(client_socket,))
    receive_thread.start()
    
    # 主线程负责发送音频数据
    send_audio_data(client_socket, 'tests/data/asr_example_zh.wav')
    
    # 等待接收线程完成
    receive_thread.join()