import numpy as np
import gym
from gym import spaces
from gym_subgoal_automata.utils import utils


class GridWorldActions:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class GridWorldEnv(gym.Env):
    HIDE_STATE_VARIABLES = "hide_state_variables"
    RANDOM_SEED_FIELD = "seed"
    USE_ONE_HOT_VECTOR_FIELD = "use_one_hot_vector_states"

    def __init__(self, params=None):
        super(GridWorldEnv, self).__init__()

        self.params = params

        # game properties
        self.is_game_over = False
        self.seed = utils.get_param(self.params, GridWorldEnv.RANDOM_SEED_FIELD)

        # state properties
        self.hide_state_variables = utils.get_param(self.params, GridWorldEnv.HIDE_STATE_VARIABLES, False)
        self.use_one_hot_vector = utils.get_param(self.params, GridWorldEnv.USE_ONE_HOT_VECTOR_FIELD, False)

        self.action_space = spaces.Discrete(4)

    def step(self, action):
        raise NotImplementedError

    def get_state_id(self, num_states, state_possible_values, state_variables):
        cell_index = 0
        content_index = num_states

        for i in range(0, len(state_possible_values)):
            content_index /= state_possible_values[i]
            cell_index += state_variables[i] * content_index

        return int(cell_index)

    def get_one_hot_state(self, num_states, state_id):
        state = np.zeros(num_states, dtype=np.float32)
        state[state_id] = 1.0
        return state

    def reset(self):
        self.is_game_over = False
        return None

    def render(self, mode='human'):
        raise NotImplementedError

    def play(self):
        self.reset()
        self.render()

        total_reward = 0.0
        is_done = False

        while not is_done:
            a = input("Enter an action: ")

            action = None
            if a == "w":
                action = GridWorldActions.UP
            elif a == "s":
                action = GridWorldActions.DOWN
            elif a == "a":
                action = GridWorldActions.LEFT
            elif a == "d":
                action = GridWorldActions.RIGHT

            if action is not None:
                _, reward, is_done, _ = self.step(action)
                total_reward += reward
                self.render()
            else:
                print("Invalid action: use 'w' (up), 's' (down), 'a' (left) or 'd' (right).")

        print("Game finished. Total reward: %.2f." % total_reward)
