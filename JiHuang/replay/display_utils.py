#-*- coding: utf-8 -*-

import os
import curses
import numpy as np

STYLE = {
    'fore':
    {   # 前景色
        'black'    : 30,   #  黑色
        'red'      : 31,   #  红色
        'green'    : 32,   #  绿色
        'yellow'   : 33,   #  黄色
        'blue'     : 34,   #  蓝色
        'purple'   : 35,   #  紫红色
        'cyan'     : 36,   #  青蓝色
        'white'    : 37,   #  白色
    },

    'back' :
    {   # 背景
        'black'     : 40,  #  黑色
        'red'       : 41,  #  红色
        'green'     : 42,  #  绿色
        'yellow'    : 43,  #  黄色
        'blue'      : 44,  #  蓝色
        'purple'    : 45,  #  紫红色
        'cyan'      : 46,  #  青蓝色
        'white'     : 47,  #  白色
    },

    'mode' :
    {   # 显示模式
        'mormal'    : 0,   #  终端默认设置
        'bold'      : 1,   #  高亮显示
        'half'      : 2,   #  半高亮显示
        'italic'    : 3,   #  斜体
        'underline' : 4,   #  使用下划线
        'blink'     : 5,   #  闪烁，6也是闪烁
        'invert'    : 7,   #  反白显示
        'hide'      : 8,   #  不可见
        'strikeout' : 9,   #  删除线
    },

    'default' :
    {
        'end' : 0,
    },
}
record = {
    '0' : 0,
    '1' : 1,
    '2' : 2,
    '3' : 3,
    '10004' : 4,
    '10005' : 5,
    '10006' : 6,
}

def as_style(string, mode = '', fore = '', back = ''): 
    mode = STYLE['mode'][mode] if mode in STYLE['mode'] else ''
    fore = STYLE['fore'][fore] if fore in STYLE['fore'] else ''
    back = STYLE['back'][back] if back in STYLE['back'] else ''
    style = ';'.join([str(mode), str(fore), str(back)])
    style = '\033[%sm' % style
    end   = '\033[%sm' % STYLE['default']['end']
    return '%s%s%s' % (style, string, end)

def get_code(type_id):
    return record[str(type_id)]

def init_win():
    global screen
    screen = curses.initscr()
    curses.start_color()
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_BLUE, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_GREEN)
    curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_BLUE)
    curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_YELLOW)
    curses.noecho()


def close_win():
    curses.endwin()

def input_key():
    global screen
    return screen.getkey()

def display_map(env, step):
    global screen
    # name = [' ', 'A', 'P', 'B', 'T', 'R', 'M']
    name = [' ', 'A', 'P', 'B', ' ', ' ', ' ']

    info = np.array(env.get_map_information()).astype(np.int)
    _map = np.zeros(shape=(info[0], info[1])).astype(np.int)
    
    idx = 2 + 2 * info[0] * info[1]
    agent_num = info[idx]
    idx += 1
    for agent in range(agent_num):
        _map[info[idx], info[idx+1]] = get_code(1)
        idx += 2
    
    animal_num = info[idx]
    idx += 1
    for animal in range(animal_num):
        _map[info[idx], info[idx+1]] = get_code(info[idx+4])
        _map[info[idx+2], info[idx+3]] = 3
        idx += 5
    
    plant_num = info[idx]
    idx += 1
    for plant in range(plant_num):
        _map[info[idx], info[idx+1]] = get_code(info[idx+2])
        idx += 3
    
    resource_num = info[idx]
    idx += 1
    for resource in range(resource_num):
        _map[info[idx], info[idx+1]] = get_code(info[idx+2])
        idx += 3

    # idx = 2
    # for x in range(info[0]):
    #     for y in range(info[1]):
    #         _map[x, y] += float(info[idx+1]) / 100
    #         idx += 2
    
    # lines = [(f'pig num is {animal_num}'), f'step : {step:d}']
    # # print(f'step : {step:d}')
    # for i in range(_map.shape[0]):
    #     # print(''.join([as_style(name[int(c)], mode=mode[int(c)], fore=fore[int(c)], back=back[int((c - int(c)) * 100)]) for c in _map[i, :]]))
    #     lines.append(''.join([name[int(c)] for c in _map[i, :]]))

    
    H = os.get_terminal_size().lines
    W = os.get_terminal_size().columns
    if H < 2:
        return
    screen.addstr(0, 0, f'step : {step:d}')
    for x in range(info[0]):
        for y in range(info[1]):
            screen.addstr(x + 1, y, name[_map[x, y]], curses.color_pair(_map[x, y]))
            if y >= W - 1:
                break
        if x >= H - 2:
            break
    screen.addstr(H - 1, 0, "[q]:quit [r]:reset [other]:continue")
    screen.refresh()