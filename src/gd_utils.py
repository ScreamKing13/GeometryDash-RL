import pyautogui as pygui
from os.path import join, normpath
from collections import namedtuple
import cv2
import time
import numpy as np
import mss
from keyboard import PressKey, ReleaseKey
from random import choice

Region = namedtuple("Region", ['x', 'y', 'width', 'height'])
res_dir = normpath(r"C:\Users\screm\PycharmProjects\GD_RL\res")


class GDenv:
    def __init__(self, resolution):
        self.region_corner = Region(*pygui.locateOnScreen(join(res_dir, "gd_upper.png")))
        self.region_main = Region(self.region_corner.x, self.region_corner.y + self.region_corner.height, *resolution)
        self.region_main = dict(zip(('left', 'top', 'width', 'height'), [*self.region_main]))
        pygui.click(self.region_corner)
        self.retry_img = cv2.imread(join(res_dir, "retry.png"))
        self.retry_pos = (self.region_main['left'] + 230, self.region_main['top'] + 500)
        self.fr = 0
        self.record_frame = None

    def retry(self):
        self.fr = 0
        pygui.click(self.retry_pos)

    @staticmethod
    def jump():
        PressKey()
        ReleaseKey()

    def level_failed(self, frame):
        return pygui.locate(self.retry_img, frame[453:547, 188: 278], confidence=0.9) is not None

    def step(self, action):
        if action == 1:
            self.jump()
        with mss.mss() as sct:
            frame = np.array(sct.grab(self.region_main), dtype=np.uint8)
            self.record_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            new_state = cv2.cvtColor(frame[0:554, 246:800], cv2.COLOR_BGRA2RGB)
            new_state = cv2.resize(new_state, (120, 120))
        done = self.level_failed(self.record_frame)
        reward = -100 if done else 0
        self.fr += 1

        return new_state, reward, done, self.fr

    def test(self, a):
        frames = []
        self.retry()
        last_time = time.time()
        done = False
        fr = 0
        while not done:
            # 800x600 windowed mode
            res, reward, done, fr = self.step(choice([0, 1]))
            frames.append(self.record_frame)
            fr += 1
            print(f'reward: {reward}, done: {done}')
        ex_time = time.time() - last_time
        print(f'Ex time: {ex_time}, frames: {fr}, framerate: {float(fr / ex_time)}')

        height, width, _ = frames[0].shape
        output = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, (width, height))
        for frame in frames:
            output.write(frame)
        output.release()


if __name__ == "__main__":
    e = GDenv((800, 600))
    e.test(1)
    # print(np.array(cv2.imread("test/fr_1.png", 0)))
    # e.retry()
    # done = False
    # i = 0
    # while not done:
        # with mss.mss() as sct:
        #     f = cv2.cvtColor(np.array(sct.grab(e.region_main)), cv2.COLOR_BGRA2RGB)
        #     # print(f)
        #     # print(e.retry_img)
        #     cv2.imwrite(f"test/{i}.png", f[127:459, 238:570])
        #     i += 1
        #     done = e.level_failed(f)
    # print(pygui.locate("../res/retry.png", "test/251.png"))
    # im = cv2.imread("test/251.png")[453:547, 188: 278]
    # cv2.imshow('image', im)
    # cv2.waitKey(0)
    #
    # # closing all open windows
    # cv2.destroyAllWindows()