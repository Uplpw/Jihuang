import sys
sys.path.insert(-1, "/home/vcis2/Userlist/Lipengwei/isrc/JiHuang/python")
import gym, Jihuang


if __name__ == '__main__':
    e = gym.make('jihuang-v0')
    e.reset()
    print("action_space:", e.action_space)
    # print(e.action_space.high[0])
    for _ in range(1):
        obs, reward, done, info = e.step([[0,0,0,0]])
        print(obs[0], len(obs[0]))