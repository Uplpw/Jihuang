#!/usr/bin/python3
import argparse
import os
import sys
sys.path.insert(-1, "/root/JiHuang/python")
#sys.path.insert(-1, "/lustre/S/guojiaming/up2date/ZhuoZhu/python")
import jihuang._jihuang as game
import random
import socket
import math
import struct

# HOST='120.236.247.203'
# PORT=32006
HOST='0.0.0.0'
PORT=60001
BUFFER_SIZE = 1024
HEAD_STRUCT = '128sIq'

hostname = socket.gethostname()
ipaddr = socket.gethostbyname(hostname)
print("IP address : ", ipaddr)

# 定义socket类型，网络通信，TCP
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 连接的IP与端口
server_socket.connect((HOST, PORT))



def synchronize():
    print("synchronize")
    data = ''
    while True:
        recv = server_socket.recv(BUFFER_SIZE)
        data = data + recv.decode('utf-8')
        if len(data) == 0:
            continue
        elif data[-1] == "%":
            break
    return data


def get_file_info(file_path):
    file_name = os.path.basename(file_path)
    file_name_len = len(file_name)
    file_size = os.path.getsize(file_path)
    return file_name, file_name_len, file_size



def send_file(file_path):
    file_name, file_name_len, file_size = get_file_info(file_path)
    file_head = struct.pack(HEAD_STRUCT, file_name.encode('utf-8'), file_name_len, file_size)

    server_socket.send(file_head)
    sent_size = 0

    with open(file_path) as fr:
        while sent_size < file_size:
            s_file = fr.read(BUFFER_SIZE)
            sent_size += BUFFER_SIZE
            server_socket.send(s_file.encode('utf-8'))



if __name__ == "__main__":
    # initial env
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs", help="log_dir")
    parser.add_argument("--v", type=int, default=0, help="log_level")
    parser.add_argument("--log_name", type=str,
                        default="py_jihuang", help="log_name")
    parser.add_argument("--env_param", type=str,
                        default="/root/JiHuang/examples/jihuang/example_season_32.prototxt", help="env param file")
    parser.add_argument("--env_config", type=str,
                        default="/root/JiHuang/examples/jihuang_env_config/config_season_32.prototxt", help="env config file")
    args = parser.parse_args()

    env_param = args.env_param
    env_config = args.env_config
    log_dir = args.log_dir
    log_name = args.log_name
    log_level = args.v
    env = game.Env(env_param, env_config, log_dir, log_name, log_level)

    # send config file
    print(" ####### Ready Send Config File ####### ")
    synchronize()
    send_file(env_config)
    print(" ####### Send Config File Complete ####### ")

    # send initial info
    select = 0
    if select == 0:
        # send initial map
        init_param = env.get_initialize_map()
        init_param = [int(i) for i in init_param]
        init_param = [str(i) for i in init_param]
        init_str = "#".join(init_param) + "#*"
        print(" ####### Ready Send Initialize Information ####### ")
        synchronize()
        server_socket.send(init_str.encode('utf8'))
        # print(init_str)
        print(" ####### Send Initialize Information Complete ####### ")


    # initial agent
    obs = env.get_agent_observe()
    for i in range(1, 10000000):
        env_action = [[0, 0, 0, 0]]
        env.step(env_action)

        # observe
        obs = env.get_agent_observe()

        ## show information
        param = env.get_show_environment_information()
        param = [str(i) for i in param]
        param_str = "#".join(param) + "#*"

        print(" ******** Ready Send Show Info ******** ")
        synchronize()
        server_socket.send(param_str.encode('utf8'))
        #print(param_str)
        print(" ******** Send Show Info Complete ******** ")

        # move information
        param_move = env.get_move_information()
        param_move = [str(i) for i in param_move]
        param_move_str = "#".join(param_move) + "#*"


        print(" ******** Ready Send Move Info ******** ")
        synchronize()
        server_socket.send(param_move_str.encode('utf8'))
        print(" ******** Send Move Info Complete ******** ")  
