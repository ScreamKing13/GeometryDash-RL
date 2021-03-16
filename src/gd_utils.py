import pyautogui as pygui
from os.path import join, normpath
from collections import namedtuple
import cv2
import time
import numpy as np
import mss
from keyboard_my import PressKey, ReleaseKey
import keyboard
import tensorflow as tf

Region = namedtuple("Region", ['x', 'y', 'width', 'height'])
res_dir = normpath(r"C:\Users\screm\PycharmProjects\GD_RL\res")


class GDenv:
    def __init__(self, resolution, mode="usual"):
        self.region_corner = Region(*pygui.locateOnScreen(join(res_dir, "gd_upper_new.png")))
        self.region_main = Region(self.region_corner.x, self.region_corner.y + self.region_corner.height, *resolution)
        self.region_main = dict(zip(('left', 'top', 'width', 'height'), [*self.region_main]))
        pygui.click(self.region_corner)
        self.retry_img = cv2.imread(join(res_dir, "retry.png"))
        self.progress_bar_img = cv2.cvtColor(cv2.imread(join(res_dir, "progress_bar.png")), cv2.COLOR_RGB2GRAY)
        self.retry_pos = (self.region_main['left'] + 230, self.region_main['top'] + 500)
        self.fr = 0
        self.record_frame = None
        self.pressed = False
        self.mode = mode
        self.reached = True
        if mode == "practice":
            self.practice_pos = (self.region_main['left'] + 230, self.region_main['top'] + 350)
            self.attempt_counter = 0
            MODEL_NAME = 'C:\\Users\\screm\\PycharmProjects\\GD_RL\\src\\detection_model\\result_model'
            PATH_TO_CKPT = MODEL_NAME + '\\frozen_inference_graph.pb'
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.detection_sess = tf.Session(graph=detection_graph, config=config)
            self.agent_absent = []
            self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            self.scores = detection_graph.get_tensor_by_name('detection_scores:0')

    def start_practice(self):
        PressKey(0x39, False)
        ReleaseKey(0x39, False)
        time.sleep(2)
        PressKey(0x01, False)
        ReleaseKey(0x01, False)
        time.sleep(1)
        pygui.click(self.region_corner)
        time.sleep(1)
        pygui.click(self.practice_pos)
        time.sleep(0.25)

    def retry(self):
        self.fr = 0
        if self.mode == "practice":
            self.agent_absent = [False for _ in range(10)]
            with mss.mss() as sct:
                if self.reached:
                    self.attempt_counter = 0
                    restart_visible = False
                    while not restart_visible:
                        time.sleep(0.5)
                        frame = np.array(sct.grab(self.region_main), dtype=np.uint8)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        restart_visible = self.level_failed(frame)
                    self.start_practice()
                else:
                    self.attempt_counter += 1
                    appeared = False
                    t = time.perf_counter()
                    while not appeared and (time.perf_counter() - t) < 7:
                        frame = np.array(sct.grab(self.region_main), dtype=np.uint8)
                        image_np = np.array(cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB))
                        image_np = np.expand_dims(image_np, axis=0)
                        scores = self.detection_sess.run(
                            self.scores,
                            feed_dict={self.image_tensor: image_np})
                        if scores.max() >= 0.98:
                            appeared = True
                    if not appeared:
                        self.uncheckpoint()
        else:
            PressKey(0x39, False)
            ReleaseKey(0x39, False)
        self.reached = False

    def level_failed(self, frame):
        return pygui.locate(self.retry_img, frame[453:547, 188: 278], confidence=0.9) is not None

    def reached_heaven(self, frame):
        return pygui.locate(self.progress_bar_img,
                            cv2.cvtColor(frame[4:22, 226:574], cv2.COLOR_BGR2GRAY), confidence=0.99) is not None

    def step(self, action):
        if not self.pressed and action == 1:
            PressKey()
            self.pressed = True
        elif self.pressed and action == 0:
            ReleaseKey()
            self.pressed = False
        with mss.mss() as sct:
            frame = np.array(sct.grab(self.region_main), dtype=np.uint8)
            self.record_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if self.mode == "practice":
                image_np = np.array(cv2.cvtColor(self.record_frame[22:, :], cv2.COLOR_BGR2RGB))
                image_np = np.expand_dims(image_np, axis=0)
                scores = self.detection_sess.run(
                    self.scores,
                    feed_dict={self.image_tensor: image_np})
            new_state = cv2.cvtColor(frame[22:, :], cv2.COLOR_BGRA2RGB)
            new_state = cv2.resize(new_state, (84, 84))
        self.reached |= self.reached_heaven(self.record_frame)
        if self.mode == "practice":
            if scores.max() < 0.98:
                self.agent_absent.append(True)
            else:
                self.agent_absent.append(False)
            self.agent_absent.pop(0)
            done = True if all(self.agent_absent) else False
        else:
            done = self.level_failed(self.record_frame)
        reward = 0
        if done:
            if self.pressed:
                ReleaseKey()
            if self.reached:
                reward = 0
            else:
                reward = -100
                self.uncheckpoint()
                self.uncheckpoint()
        self.fr += 1

        return new_state, reward, done, self.fr, self.reached, self.attempt_counter

    def pause(self):
        PressKey(0x01, False)
        ReleaseKey(0x01, False)
        time.sleep(1)

    def unpause(self):
        PressKey(0x39, False)
        ReleaseKey(0x39, False)
        time.sleep(1)

    def uncheckpoint(self):
        PressKey(0x2D, False)
        ReleaseKey(0x2D, False)

    def test(self):
        for _ in range(10):
            frames = []
            self.retry()
            last_time = time.time()
            done = False
            fr = 0
            reached = False
            while not done:
                # 800x600 windowed mode
                res, reward, done, fr, _, _ = self.step(1 if keyboard.is_pressed('W') else 0)
                frames.append(self.record_frame)
                reached |= self.reached_heaven(self.record_frame)
                fr += 1
                print(f'reward: {reward}, done: {done}, reached: {reached}')
                time.sleep(1 / 30)
            ex_time = time.time() - last_time
            print(f'Ex time: {ex_time}, frames: {fr}, framerate: {float(fr / ex_time)}')


if __name__ == "__main__":
    e = GDenv((800, 600), mode="practice")
    e.test()