import os
import sys
import pickle
import copy

sys.path.insert(-1, '/lustre/S/yuankaizhao/project/JiHuang/python')
import jihuang._jihuang as game
import numpy as np

# import game_wrapper as gw
# gwl=[i for i in range(gw.goal_num)]

from display_utils import *
from curses import wrapper

def main(stdscr):
    run_log = True if len(sys.argv) > 1 and sys.argv[1]=='T' else False
    # mode = 'day_l_goal'
    env_param = 'example_debug_40.prototxt'
    env_config = 'config_debug_40.prototxt'
    env = game.Env(env_param, env_config, '', '', 101)
    log_name = 'action_log'
    if run_log:
        env_action_list = pickle.load(open(log_name, "rb"))
    else:
        env_action_list = []
    
    init_win()
    idx = 0
    pre_struct_state = None
    while True:
        # os.system('clear')
        obs = env.get_agent_observe()[0]
        avl_action = [str(i) for i in range(50)]
        if run_log and idx < len(env_action_list):
            # print(env_action_list)
            env_action = env_action_list[idx]
            display_map(env, step=idx)
        else:
            close_win()
            while(True):
                action = input('action: ')
                if action == 'save':
                    pickle.dump(env_action_list, open(log_name, "wb"))
                    print(env_action_list)
                    sys.exit()
                elif action == 'quit':
                    sys.exit()
                elif action in avl_action:
                    env_action = [gw.get_action(int(action), struct_state)]
                    break
                else:
                    continue
            env_action_list.append(env_action)
        env.step(env_action)
        idx = idx + 1
        key = input_key()
        if key == 'q':
            close_win()
            sys.exit()
        elif key == 'r':
            idx = idx - 1
            env.reset()

if __name__ == "__main__":
    wrapper(main)