import gym
from gym import Wrapper, spaces
import cv2
import numpy as np
from collections import deque

cv2.ocl.setUseOpenCL(False)

"""
atari wrapper like environment
frame stack = 4
observation shape: (4, 84, 84)
"""


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, dict_space_key=None):
        super().__init__(env)
        self._width = width
        self._height = height
        num_colors = 1
        
        self.buffer = []
        for i in range(180):
            frame = cv2.imread(f"MountainCar_frame_buffer/{i}.jpg")
            # bgr -> rgb
            frame = frame[:, :, ::-1]
            self.buffer.append(frame)

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        self.observation_space = new_space

    def _position_to_idx(self, position):
        return int((position + 1.2) * 100)

    def observation(self, obs):
        position = obs[0]
        idx = self._position_to_idx(position)
        frame = self.buffer[idx]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        frame = np.expand_dims(frame, -1)
        obs = frame
        return obs


class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[0]

    def frame(self, i):
        return self._force()[i, ...]


class WarpPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=((shp[0] * k, ) + shp[1:]),
                                            dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


def make_mc():
    env = gym.make('MountainCar-v0')
    env = WarpFrame(env)
    env = WarpPyTorch(env)
    env = FrameStack(env, 4)

    return env


if __name__ == "__main__":
    make_mc()