from datetime import datetime as dt
import copy
import sys
import socket
import struct
import os


HOST = '0.0.0.0'
PORT = 60001
BUFSIZ = 4096
ADDRESS = (HOST, PORT)

BUFFER_SIZE=1024
HEAD_STRUCT='128sIq'

hostname = socket.gethostname()
ipaddr = socket.gethostbyname(hostname)
print("IP address : ", ipaddr)


# 创建监听socket
tcpServerSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定IP地址和固定端口
tcpServerSocket.bind(ADDRESS)
tcpServerSocket.listen()
# connect1
client_socket1, client_address1 = tcpServerSocket.accept() 
print("Connect!! client_address = ", client_address1)
# connect2
client_socket2, client_address2 = tcpServerSocket.accept() 
print("Connect!! client_address = ", client_address2)

# 收到的字符转为一个字符串
def getRecvFromClient1():
    data = ''
    while True:
        recv = client_socket1.recv(1024)
        data = data + recv.decode('utf-8')
        if len(data) == 0:
            continue
        elif data[-1] == "*":
            break
    return data

# 发送数据
def sendInfoToClient1(actions_str):
    client_socket1.send(actions_str.encode('utf8'))

# 收到的字符转为一个字符串
def getRecvFromClient2():
    data = ''
    while True:
        recv = client_socket2.recv(1024)
        data = data + recv.decode('utf-8')
        if len(data) == 0:
            continue
        elif data[-1] == "*":
            break
    return data
# 发送数据
def sendInfoToClient2(actions_str):
    client_socket2.send(actions_str.encode('utf8'))


# 同步操作
def synchronize():
    print("synchronize")
    data = ''
    while True:
        recv = client_socket2.recv(1024)
        data = data + recv.decode('utf-8')
        if len(data) == 0:
            continue
        elif data[-1] == "*":
            break
    return data

# 解压缩
def unpack_file_info(file_info):
    file_name, file_name_len, file_size = struct.unpack(HEAD_STRUCT, file_info)
    file_name = file_name[:file_name_len]
    return file_name, file_size


def get_file_info(file_path):
    file_name = os.path.basename(file_path)
    file_name_len = len(file_name)
    file_size = os.path.getsize(file_path)
    return file_name, file_name_len, file_size

# 接受文件
def recv_file():
    file_info_package = client_socket1.recv(BUFFER_SIZE)
    file_name, file_size = unpack_file_info(file_info_package)

    recved_size = 0
    with open(file_name, 'wb') as fw:
        while recved_size < file_size:
            r_file = client_socket1.recv(BUFFER_SIZE)
            client_socket2.send(r_file)
            if not len(r_file):
                break
            recved_size += BUFFER_SIZE
            fw.write(r_file)
    client_socket2.send("*".encode('utf-8'))
    # 转发
    #send_file(file_name)
    

def send_file(file_path):
    file_name, file_name_len, file_size = get_file_info(file_path)
    file_head = struct.pack(HEAD_STRUCT, file_name.encode('utf-8'), file_name_len, file_size)

    client_socket2.send(file_head)
    sent_size = 0
    with open(file_path) as fr:
        while sent_size < file_size:
            s_file = fr.read(BUFFER_SIZE)
            sent_size += BUFFER_SIZE
            client_socket2.send(s_file.encode('utf-8'))



if __name__ == "__main__":
    # client1: 内网服务器，跑神经网络
    # client2: windows端，跑虚幻4
    # 目前不需要玩家，，因此只从client1接收数据发送到client2

    # send config file
    sendInfoToClient1("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    recv_file()
    print(" ####### Send config file Complete ####### ")

    select = 0
    if select == 0:
        # initialize map
        sendInfoToClient1("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        initial_data = getRecvFromClient1()
        sendInfoToClient2(initial_data)
        print(" ####### Send Initialize Information Complete ####### ")

    while True:
        sendInfoToClient1("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        show_data = getRecvFromClient1()
        sendInfoToClient2(show_data)
        print(" ******** Send Show Info Complete ******** ")
            
        sendInfoToClient1("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        move_data = getRecvFromClient1()
        sendInfoToClient2(move_data)
        print(" ******** Send Move Info Complete ******** ")  

        # 等待 client2
        synchronize()

