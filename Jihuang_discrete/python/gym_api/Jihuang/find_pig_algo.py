import random
import math


def random_find_pig():
    offset_x = random.randint(-4, 4)
    if offset_x > 0:
        offset_y = (int)(math.sqrt(4 ** 2 - offset_x ** 2))
    else:
        offset_y = -(int)(math.sqrt(4 ** 2 - offset_x ** 2))
    # print("random_find_pig", offset_x, offset_y)
    return offset_x, offset_y


def random_constraint_find_pig():
    offset_x = random.randint(-4, 4)
    if offset_x > 0:
        offset_y = (int)(math.sqrt(4 ** 2 - offset_x ** 2))
    else:
        offset_y = -(int)(math.sqrt(4 ** 2 - offset_x ** 2))
    return offset_x, offset_y

#
# for i in range(100):
#     print(random_find_pig(), end=" ")