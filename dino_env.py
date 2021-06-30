from gym import Env, spaces

import numpy as np
import time
from PIL import ImageGrab, ImageOps, ImageFilter
import matplotlib.pyplot as plt
from random import randint

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

import config


class DinoEnv(Env):

    def __init__(self, dino_vision=True):
        super(DinoEnv, self).__init__()

        # driver setup
        self._configure_webdriver(config.webdriver_path)

        # gym env setup
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=self._get_observation_shape(),
                                            dtype=np.uint8)

        self.state = self._get_observation()

        # visualization utils
        self.dino_vision = dino_vision

    def step(self, action):
        self._take_action(action)
        observation = self._get_observation()
        done = self._get_crash()
        if done:
            reward = -1
        else:
            reward = 1
        info = {}
        return observation, reward, done, info

    def reset(self):
        self.webdriver.execute_script("Runner.instance_.restart()")
        self.state = self._get_observation()
        return self.state

    def render(self, **kwargs):
        pass
        # if self.dino_vision:
        #     plt.imshow(observation, cmap='gray')
        #     plt.show()
        #     plt.pause(0.0001)
        #     plt.clf()

    def _take_action(self, action):
        # jump
        if action == 1:
            self.webdriver.find_element_by_id("t").send_keys(Keys.ARROW_UP)
        # dodge
        if action == 2:
            self.webdriver.find_element_by_id("t").send_keys(Keys.ARROW_DOWN)
        # else do nothing

    def _get_crash(self):
        crashed = self.webdriver.execute_script("return Runner.instance_.crashed")
        return crashed

    def get_score(self):
        distance = self.webdriver.execute_script("return Runner.instance_.distanceRan")
        score = self.webdriver.execute_script(f"return Runner.instance_.distanceMeter.getActualDistance({distance})")
        return score

    def _configure_webdriver(self, webdriver_path):
        # webdriver and game setup
        self.webdriver = webdriver.Chrome(executable_path=webdriver_path)
        self.webdriver.set_window_position(-10, 0)
        self.webdriver.get('chrome://dino')
        time.sleep(1)
        # jump to start the game
        self.webdriver.find_element_by_id("t").send_keys(Keys.ARROW_UP)

    @staticmethod
    def _get_observation():
        img = ImageGrab.grab(bbox=(config.left,
                                   config.upper,
                                   config.right,
                                   config.lower))
        img = ImageOps.scale(img, config.scale)
        # img = img.resize((112, 112))
        img = ImageOps.grayscale(img)
        img = img.filter(ImageFilter.FIND_EDGES)
        img = np.asarray(img)
        return img

    @staticmethod
    def _get_observation_shape():
        height = (config.lower - config.upper) * config.scale
        width = (config.right - config.left) * config.scale
        shape = (int(height), int(width))
        return shape


if __name__ == '__main__':

    plt.ion()

    dino = DinoEnv()

    print(dino.observation_space)
    print(dino.action_space)
    print(dino.action_space.sample())

    episodes = 10

    for episode in range(episodes):
        state = dino.reset()
        done = False
        score = 0

        while not done:
            action = randint(0, 1)
            obs, reward, done, _ = dino.step(action)
            score += reward
            print(action, obs.shape, reward, score, done)

        print('Episode:{} Score:{}'.format(episode, score))
        print()
        time.sleep(2)





