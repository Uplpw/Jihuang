"""
utils file
"""
import random
import math


def list_dict_contain(list_dict, obj):
    for i in list_dict:
        if int(i[2]) == obj:
            return True
    return False


def random_find_pig():
    offset_x = random.randint(-4, 4)
    if offset_x > 0:
        offset_y = (int)(math.sqrt(4 ** 2 - offset_x ** 2))
    else:
        offset_y = -(int)(math.sqrt(4 ** 2 - offset_x ** 2))
    return offset_x, offset_y


def dictCompare(dict1, dict2, key=None):
    list_key = []
    if key == None:
        for key in dict1.keys():
            if dict1[key] != dict2[key]:
                list_key.append(key)
    else:
        if dict1[key] != dict2[key]:
            list_key.append(key)
    return list_key
