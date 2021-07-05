# -*- coding: utf-8 -*-

import os
import curses
import numpy as np


class MapDisplayer:
    def __init__(self, record=None, name=None):
        self.record = {
            # 将typeid映射成int，对应于curses.init_pair的id
            '0': 0,
            '1': 1,
            '2': 2,
            '3': 3,
            '10004': 4,
            '10005': 5,
            '10006': 6,
        } if record is None else record
        self.name = [' ', 'A', 'P', 'B', ' ', ' ', ' '] if name is None else name
        # self.name = [' ', 'A', 'P', 'B', 'T', 'R', 'M']
        try:
            self.screen = curses.initscr()
            self.screen.keypad(1)
            self.init_curses()
        except:
            self.close()

    def init_curses(self):
        curses.noecho()
        curses.cbreak()
        try:
            curses.start_color()
            curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_BLUE, curses.COLOR_BLACK)
            curses.init_pair(3, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
            curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_GREEN)
            curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_BLUE)
            curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_YELLOW)
        except:
            pass

    def close(self):
        self.screen.keypad(0)
        curses.echo()
        curses.nocbreak()
        curses.endwin()

    def input_key(self):
        return self.screen.getkey()

    def display_map(self, env, step):
        info = np.array(env.get_map_information()).astype(np.int)
        _map = np.zeros(shape=(info[0], info[1])).astype(np.int)

        idx = 2 + 2 * info[0] * info[1]
        agent_num = info[idx]
        idx += 1
        for agent in range(agent_num):
            _map[info[idx], info[idx + 1]] = self.record[str(1)]
            idx += 2

        animal_num = info[idx]
        idx += 1
        for animal in range(animal_num):
            _map[info[idx], info[idx + 1]] = self.record[str(info[idx + 4])]
            _map[info[idx + 2], info[idx + 3]] = 3
            idx += 5

        plant_num = info[idx]
        idx += 1
        for plant in range(plant_num):
            _map[info[idx], info[idx + 1]] = self.record[str(info[idx + 2])]
            idx += 3

        resource_num = info[idx]
        idx += 1
        for resource in range(resource_num):
            _map[info[idx], info[idx + 1]] = self.record[str(info[idx + 2])]
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
        self.screen.addstr(0, 0, f'step : {step:d}')
        for x in range(info[0]):
            for y in range(info[1]):
                self.screen.addstr(x + 1, y, self.name[_map[x, y]], curses.color_pair(_map[x, y]))
                if y >= W - 1:
                    break
            if x >= H - 2:
                break
        self.screen.addstr(H - 1, 0, "[q]:quit [r]:reset [other]:continue")
        self.screen.refresh()
