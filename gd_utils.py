import pyautogui as pygui
from os.path import join, normpath
from collections import namedtuple
import cv2
import time
import numpy as np
import mss
from keyboard import PressKey, ReleaseKey

Region = namedtuple("Region", ['x', 'y', 'width', 'height'])
res_dir = normpath(r"C:\Users\screm\PycharmProjects\GD_RL\res")


class GDenv:
    def __init__(self, resolution):
        self.region_corner = Region(*pygui.locateOnScreen(join(res_dir, "gd_upper.png")))
        self.region_main = Region(self.region_corner.x, self.region_corner.y + self.region_corner.height, *resolution)
        self.region_main = dict(zip(('left', 'top', 'width', 'height'), [*self.region_main]))
        pygui.click(self.region_corner)
        self.retry_img = cv2.imread(join(res_dir, "retry_gray.png"), 0)
        self.retry_pos = (self.region_main['left'] + 230, self.region_main['top'] + 500)
        self.fr = 0

    def retry(self):
        self.fr = 0
        pygui.click(self.retry_pos)

    @staticmethod
    def jump():
        PressKey()
        ReleaseKey()

    def level_failed(self, frame):
        return pygui.locate(self.retry_img, frame[91: 109, 28:42], confidence=0.9) is not None

    def step(self, action):
        if action == 1:
            self.jump()
        with mss.mss() as sct:
            new_state = np.array(sct.grab(self.region_main), dtype=np.uint8)
            new_state = cv2.cvtColor(new_state, cv2.COLOR_BGR2GRAY)
            new_state = cv2.resize(new_state, (120, 120))
        done = self.level_failed(new_state)
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
            res, reward, done = self.step(2)
            frames.append(res)
            fr += 1
            print(f'reward: {reward}, done: {done}')
        ex_time = time.time() - last_time
        print(f'Ex time: {ex_time}, frames: {fr}, framerate: {float(fr / ex_time)}')

        for i, frame in enumerate(frames):
            cv2.imwrite(f"test/fr_{i}.png", frame)
