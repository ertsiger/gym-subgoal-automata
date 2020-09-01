import random
from abc import ABC
from gym import spaces
from gym_subgoal_automata.envs.gridworld.gridworld_env import GridWorldEnv, GridWorldActions
from gym_subgoal_automata.utils.subgoal_automaton import SubgoalAutomaton
from gym_subgoal_automata.utils import utils

class CraftWorldObject:
    AGENT = "A"
    WOOD = "a"
    TOOLSHED = "b"
    WORKBENCH = "c"
    GRASS = "d"
    FACTORY = "e"
    IRON = "f"
    BRIDGE = "g"
    AXE = "h"
    WALL = "X"


class CraftWorldEnv(GridWorldEnv, ABC):
    """
    The Craft World environment
    from "Modular Multitask Reinforcement Learning with Policy Sketches" by Jacob Andreas, Dan Klein and Sergey Levine.

    The implementation is based on that from "Using Reward Machines for High-Level Task Specification and Decomposition
    in Reinforcement Learning" by Rodrigo Toro Icarte, Toryn Q. Klassen, Richard Valenzano and Sheila A. McIlraith.

    Description:
    It is a grid consisting of an agent and different labelled locations (number of locations in brackets):
        - a: wood (5)
        - b: toolshed (2)
        - c: workbench (2)
        - d: grass (5)
        - e: factory (2)
        - f: iron (5)
        - g: bridge (2)
        - h: axe (2)

    Observations:
    The state space is given by the position of the agent (39 * 39).

    Rewards:
    Different tasks (subclassed below) are defined in this environment. All of them are goal-oriented, i.e., provide
    a reward of 1 when a certain goal is achieved and 0 otherwise.
        - CraftWorldMakePlankEnv: get wood, use toolshed.
        - CraftWorldMakeStickEnv: get wood, use workbench.
        - CraftWorldMakeClothEnv: get grass, use factory.
        - CraftWorldMakeRopeEnv: get grass, use toolshed.
        - CraftWorldMakeBridgeEnv: get iron, get wood, use factory (the iron and wood can be gotten in any order).
        - CraftWorldMakeBedEnv: get wood, use toolshed, get grass, use workbench (the grass can be gotten at any time before using the workbench).
        - CraftWorldMakeAxeEnv: get wood, use workbench, get iron, use toolshed (the iron can be gotten at any time before using the toolshed).
        - CraftWorldMakeShearsEnv: get wood, use workbench, get iron, use workbench (the iron can be gotten at any time before using the workbench).
        - CraftWorldGetGoldEnv: get iron, get wood, use factory, use bridge (the iron and wood can be gotten in any order).
        - CraftWorldGetGemEnv: get wood, use workbench, get iron, use toolshed, use axe (the iron can be gotten at any time before using the toolshed).

    Actions:
    There are 4 deterministic actions:
        - 0: up
        - 1: down
        - 2: left
        - 3: right

    Map generation:
    The observables and the agent can be placed anywhere in the grid. It is possible to observe more than one observable
    at the same time (e.g., wood and toolshed). The only restriction is that two observables of the same type cannot be
    placed in the same location.

    Acknowledgments:
    Most of the code has been reused from the original implementation by the authors of reward machines:
    https://bitbucket.org/RToroIcarte/qrm/src/master/.
    """
    GRID_HEIGHT = "height"
    GRID_WIDTH = "width"
    ENFORCE_SINGLE_OBSERVARBLE_PER_LOCATION = "enforce_single_observable_per_location"

    def __init__(self, params=None):
        super().__init__(params)

        self.agent = None       # agent's location
        self.prev_agent = None  # previous agent location
        self.init_agent = None  # agent's initial position, for resetting

        self.locations = {}     # locations of the objects

        # grid size
        self.height = utils.get_param(params, CraftWorldEnv.GRID_HEIGHT, 39)
        self.width = utils.get_param(params, CraftWorldEnv.GRID_WIDTH, 39)
        self.observation_space = spaces.Discrete(self._get_num_states())

        self.enforce_single_observable_per_location = utils.get_param(params, CraftWorldEnv.ENFORCE_SINGLE_OBSERVARBLE_PER_LOCATION, True)

        # the completion of the task is checked against the ground truth automaton
        self.automaton = self.get_automaton()
        self.automaton_state = None

        self._load_map()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if self.is_game_over:
            return self._get_state(), 0.0, True, self.get_observations()

        target_x, target_y = self.agent

        if action == GridWorldActions.UP:
            target_y += 1
        elif action == GridWorldActions.DOWN:
            target_y -= 1
        elif action == GridWorldActions.LEFT:
            target_x -= 1
        elif action == GridWorldActions.RIGHT:
            target_x += 1

        target_pos = (target_x, target_y)
        if self._is_valid_position(target_pos):
            self.prev_agent = self.agent
            self.agent = target_pos
            self._update_state()

        reward = 1.0 if self.is_goal_achieved() else 0.0
        self.is_game_over = self.is_terminal()

        return self._get_state(), reward, self.is_game_over, self.get_observations()

    def _get_num_states(self):
        return self.width * self.height

    def _get_state(self):
        num_states = self._get_num_states()

        state_possible_values = [self.width, self.height]
        state_variables = [self.agent[0], self.agent[1]]

        state_id = self.get_state_id(num_states, state_possible_values, state_variables)

        if self.use_one_hot_vector:
            return self.get_one_hot_state(num_states, state_id)

        return state_id

    def get_observables(self):
        return [CraftWorldObject.WOOD, CraftWorldObject.TOOLSHED, CraftWorldObject.WORKBENCH, CraftWorldObject.GRASS,
                CraftWorldObject.FACTORY, CraftWorldObject.IRON, CraftWorldObject.BRIDGE, CraftWorldObject.AXE]

    def get_observations(self):
        return set(self.locations[self.agent]) if self.agent in self.locations else {}

    def is_terminal(self):
        return self.automaton.is_terminal_state(self.automaton_state)

    def is_goal_achieved(self):
        return self.automaton.is_accept_state(self.automaton_state)

    def reset(self):
        super().reset()

        # set initial state
        self.agent = self.init_agent
        self.prev_agent = None
        self.automaton_state = self.automaton.get_initial_state()

        # update initial automaton state according to the map layout
        self._update_state()

        return self._get_state()

    def _update_state(self):
        if self.prev_agent != self.agent:
            self.automaton_state = self.automaton.get_next_state(self.automaton_state, self.get_observations())

    def _load_map(self):
        random_gen = random.Random(self.seed)

        self.init_agent = self._generate_random_pos(random_gen)
        self._add_location(CraftWorldObject.WOOD, 5, random_gen)
        self._add_location(CraftWorldObject.TOOLSHED, 2, random_gen)
        self._add_location(CraftWorldObject.WORKBENCH, 2, random_gen)
        self._add_location(CraftWorldObject.GRASS, 5, random_gen)
        self._add_location(CraftWorldObject.FACTORY, 2, random_gen)
        self._add_location(CraftWorldObject.IRON, 5, random_gen)
        self._add_location(CraftWorldObject.BRIDGE, 2, random_gen)
        self._add_location(CraftWorldObject.AXE, 2, random_gen)

    def _add_location(self, symbol, number, random_gen):
        for _ in range(number):
            pos = (-1, -1)
            while not self._is_valid_position(pos) or \
                    (pos in self.locations and symbol in self.locations[pos]) or \
                    (self.enforce_single_observable_per_location and pos in self.locations):
                pos = self._generate_random_pos(random_gen)
            if pos not in self.locations:
                self.locations[pos] = []
            self.locations[pos].append(symbol)

    def _generate_random_pos(self, random_gen):
        return random_gen.randint(0, self.width - 1), random_gen.randint(0, self.height - 1)

    def _is_valid_position(self, pos):
        return 0 <= pos[0] < self.width and 0 <= pos[1] < self.height

    def render(self, mode='human'):
        self._render_horizontal_border()

        for y in range(self.height - 1, -1, -1):
            print(CraftWorldObject.WALL, end="")
            for x in range(self.width):
                position = (x, y)
                if position == self.agent:
                    print(CraftWorldObject.AGENT, end="")
                elif position in self.locations:
                    print(self.locations[position][0], end="")  # just print one of the items in the position
                else:
                    print(" ", end="")
            print(CraftWorldObject.WALL)

        self._render_horizontal_border()

    def _render_horizontal_border(self):
        for i in range(self.width + 2):
            print(CraftWorldObject.WALL, end="")
        print()


class CraftWorldMakePlankEnv(CraftWorldEnv):
    """
    Get wood, use toolshed.
    """
    def get_restricted_observables(self):
        return [CraftWorldObject.WOOD, CraftWorldObject.TOOLSHED]

    def get_automaton(self):
        automaton = SubgoalAutomaton()
        automaton.add_state("u0")
        automaton.add_state("u1")
        automaton.add_state("u_acc")

        if self.enforce_single_observable_per_location:
            automaton.add_edge("u0", "u1", [CraftWorldObject.WOOD])
        else:
            automaton.add_edge("u0", "u1", [CraftWorldObject.WOOD, "~" + CraftWorldObject.TOOLSHED])
            automaton.add_edge("u0", "u_acc", [CraftWorldObject.WOOD, CraftWorldObject.TOOLSHED])
        automaton.add_edge("u1", "u_acc", [CraftWorldObject.TOOLSHED])

        automaton.set_initial_state("u0")
        automaton.set_accept_state("u_acc")
        return automaton


class CraftWorldMakeStickEnv(CraftWorldEnv):
    """
    Get wood, use workbench.
    """
    def get_restricted_observables(self):
        return [CraftWorldObject.WOOD, CraftWorldObject.WORKBENCH]

    def get_automaton(self):
        automaton = SubgoalAutomaton()
        automaton.add_state("u0")
        automaton.add_state("u1")
        automaton.add_state("u_acc")

        if self.enforce_single_observable_per_location:
            automaton.add_edge("u0", "u1", [CraftWorldObject.WOOD])
        else:
            automaton.add_edge("u0", "u1", [CraftWorldObject.WOOD, "~" + CraftWorldObject.WORKBENCH])
            automaton.add_edge("u0", "u_acc", [CraftWorldObject.WOOD, CraftWorldObject.WORKBENCH])
        automaton.add_edge("u1", "u_acc", [CraftWorldObject.WORKBENCH])

        automaton.set_initial_state("u0")
        automaton.set_accept_state("u_acc")
        return automaton


class CraftWorldMakeClothEnv(CraftWorldEnv):
    """
    Get grass, use factory.
    """
    def get_restricted_observables(self):
        return [CraftWorldObject.GRASS, CraftWorldObject.FACTORY]

    def get_automaton(self):
        automaton = SubgoalAutomaton()
        automaton.add_state("u0")
        automaton.add_state("u1")
        automaton.add_state("u_acc")

        if self.enforce_single_observable_per_location:
            automaton.add_edge("u0", "u1", [CraftWorldObject.GRASS])
        else:
            automaton.add_edge("u0", "u1", [CraftWorldObject.GRASS, "~" + CraftWorldObject.FACTORY])
            automaton.add_edge("u0", "u_acc", [CraftWorldObject.GRASS, CraftWorldObject.FACTORY])
        automaton.add_edge("u1", "u_acc", [CraftWorldObject.FACTORY])

        automaton.set_initial_state("u0")
        automaton.set_accept_state("u_acc")
        return automaton


class CraftWorldMakeRopeEnv(CraftWorldEnv):
    """
    Get grass, use toolshed.
    """
    def get_restricted_observables(self):
        return [CraftWorldObject.GRASS, CraftWorldObject.TOOLSHED]

    def get_automaton(self):
        automaton = SubgoalAutomaton()
        automaton.add_state("u0")
        automaton.add_state("u1")
        automaton.add_state("u_acc")

        if self.enforce_single_observable_per_location:
            automaton.add_edge("u0", "u1", [CraftWorldObject.GRASS])
        else:
            automaton.add_edge("u0", "u1", [CraftWorldObject.GRASS, "~" + CraftWorldObject.TOOLSHED])
            automaton.add_edge("u0", "u_acc", [CraftWorldObject.GRASS, CraftWorldObject.TOOLSHED])
        automaton.add_edge("u1", "u_acc", [CraftWorldObject.TOOLSHED])

        automaton.set_initial_state("u0")
        automaton.set_accept_state("u_acc")
        return automaton


class CraftWorldMakeBridgeEnv(CraftWorldEnv):
    """
    Get iron, get wood, use factory (the iron and wood can be gotten in any order).
    """
    def get_restricted_observables(self):
        return [CraftWorldObject.IRON, CraftWorldObject.WOOD, CraftWorldObject.FACTORY]

    def get_automaton(self):
        automaton = SubgoalAutomaton()
        automaton.add_state("u0")
        automaton.add_state("u1")
        automaton.add_state("u2")
        automaton.add_state("u3")
        automaton.add_state("u_acc")

        if self.enforce_single_observable_per_location:
            automaton.add_edge("u0", "u1", [CraftWorldObject.IRON, "~" + CraftWorldObject.WOOD])
            automaton.add_edge("u0", "u2", [CraftWorldObject.WOOD])
            automaton.add_edge("u1", "u3", [CraftWorldObject.WOOD])
            automaton.add_edge("u2", "u3", [CraftWorldObject.IRON])
        else:
            automaton.add_edge("u0", "u1", [CraftWorldObject.IRON, "~" + CraftWorldObject.WOOD])
            automaton.add_edge("u0", "u2", ["~" + CraftWorldObject.IRON, CraftWorldObject.WOOD])
            automaton.add_edge("u0", "u3", [CraftWorldObject.IRON, CraftWorldObject.WOOD, "~" + CraftWorldObject.FACTORY])
            automaton.add_edge("u0", "u_acc", [CraftWorldObject.IRON, CraftWorldObject.WOOD, CraftWorldObject.FACTORY])
            automaton.add_edge("u1", "u3", [CraftWorldObject.WOOD, "~" + CraftWorldObject.FACTORY])
            automaton.add_edge("u1", "u_acc", [CraftWorldObject.WOOD, CraftWorldObject.FACTORY])
            automaton.add_edge("u2", "u3", [CraftWorldObject.IRON, "~" + CraftWorldObject.FACTORY])
            automaton.add_edge("u2", "u_acc", [CraftWorldObject.IRON, CraftWorldObject.FACTORY])
        automaton.add_edge("u3", "u_acc", [CraftWorldObject.FACTORY])

        automaton.set_initial_state("u0")
        automaton.set_accept_state("u_acc")
        return automaton


class CraftWorldMakeBedEnv(CraftWorldEnv):
    """
    Get wood, use toolshed, get grass, use workbench (the grass can be gotten at any time before using the workbench).
    """
    def get_restricted_observables(self):
        return [CraftWorldObject.WOOD, CraftWorldObject.TOOLSHED, CraftWorldObject.GRASS, CraftWorldObject.WORKBENCH]

    def get_automaton(self):
        automaton = SubgoalAutomaton()
        automaton.add_state("u0")
        automaton.add_state("u1")
        automaton.add_state("u2")
        automaton.add_state("u3")
        automaton.add_state("u4")
        automaton.add_state("u5")
        automaton.add_state("u_acc")

        if self.enforce_single_observable_per_location:
            automaton.add_edge("u0", "u1", [CraftWorldObject.WOOD, "~" + CraftWorldObject.GRASS])
            automaton.add_edge("u0", "u4", [CraftWorldObject.GRASS])
            automaton.add_edge("u1", "u3", [CraftWorldObject.TOOLSHED, "~" + CraftWorldObject.GRASS])
            automaton.add_edge("u1", "u5", [CraftWorldObject.GRASS])
            automaton.add_edge("u3", "u2", [CraftWorldObject.GRASS])
            automaton.add_edge("u4", "u5", [CraftWorldObject.WOOD])
            automaton.add_edge("u5", "u2", [CraftWorldObject.TOOLSHED])
        else:
            automaton.add_edge("u0", "u1", [CraftWorldObject.WOOD, "~" + CraftWorldObject.TOOLSHED, "~" + CraftWorldObject.GRASS])
            automaton.add_edge("u0", "u2", [CraftWorldObject.WOOD, CraftWorldObject.TOOLSHED, "~" + CraftWorldObject.WORKBENCH, CraftWorldObject.GRASS])
            automaton.add_edge("u0", "u3", [CraftWorldObject.WOOD, CraftWorldObject.TOOLSHED, "~" + CraftWorldObject.GRASS])
            automaton.add_edge("u0", "u4", ["~" + CraftWorldObject.WOOD, CraftWorldObject.GRASS])
            automaton.add_edge("u0", "u5", [CraftWorldObject.WOOD, "~" + CraftWorldObject.TOOLSHED, CraftWorldObject.GRASS])
            automaton.add_edge("u0", "u_acc", [CraftWorldObject.WOOD, CraftWorldObject.TOOLSHED, CraftWorldObject.WORKBENCH, CraftWorldObject.GRASS])
            automaton.add_edge("u1", "u2", [CraftWorldObject.TOOLSHED, "~" + CraftWorldObject.WORKBENCH, CraftWorldObject.GRASS])
            automaton.add_edge("u1", "u3", [CraftWorldObject.TOOLSHED, "~" + CraftWorldObject.GRASS])
            automaton.add_edge("u1", "u5", ["~" + CraftWorldObject.TOOLSHED, CraftWorldObject.GRASS])
            automaton.add_edge("u1", "u_acc", [CraftWorldObject.TOOLSHED, CraftWorldObject.WORKBENCH, CraftWorldObject.GRASS])
            automaton.add_edge("u3", "u2", ["~" + CraftWorldObject.WORKBENCH, CraftWorldObject.GRASS])
            automaton.add_edge("u3", "u_acc", [CraftWorldObject.WORKBENCH, CraftWorldObject.GRASS])
            automaton.add_edge("u4", "u2", [CraftWorldObject.WOOD, CraftWorldObject.TOOLSHED, "~" + CraftWorldObject.WORKBENCH])
            automaton.add_edge("u4", "u5", [CraftWorldObject.WOOD, "~" + CraftWorldObject.TOOLSHED])
            automaton.add_edge("u4", "u_acc", [CraftWorldObject.WOOD, CraftWorldObject.TOOLSHED, CraftWorldObject.WORKBENCH])
            automaton.add_edge("u5", "u2", [CraftWorldObject.TOOLSHED, "~" + CraftWorldObject.WORKBENCH])
            automaton.add_edge("u5", "u_acc", [CraftWorldObject.TOOLSHED, CraftWorldObject.WORKBENCH])
        automaton.add_edge("u2", "u_acc", [CraftWorldObject.WORKBENCH])

        automaton.set_initial_state("u0")
        automaton.set_accept_state("u_acc")
        return automaton


class CraftWorldMakeAxeEnv(CraftWorldEnv):
    """
    Get wood, use workbench, get iron, use toolshed (the iron can be gotten at any time before using the toolshed).
    """
    def get_restricted_observables(self):
        return [CraftWorldObject.WOOD, CraftWorldObject.WORKBENCH, CraftWorldObject.IRON, CraftWorldObject.TOOLSHED]

    def get_automaton(self):
        automaton = SubgoalAutomaton()
        automaton.add_state("u0")
        automaton.add_state("u1")
        automaton.add_state("u2")
        automaton.add_state("u3")
        automaton.add_state("u4")
        automaton.add_state("u5")
        automaton.add_state("u_acc")

        if self.enforce_single_observable_per_location:
            automaton.add_edge("u0", "u1", [CraftWorldObject.WOOD, "~" + CraftWorldObject.IRON])
            automaton.add_edge("u0", "u4", [CraftWorldObject.IRON])
            automaton.add_edge("u1", "u3", [CraftWorldObject.WORKBENCH, "~" + CraftWorldObject.IRON])
            automaton.add_edge("u1", "u5", [CraftWorldObject.IRON])
            automaton.add_edge("u3", "u2", [CraftWorldObject.IRON])
            automaton.add_edge("u4", "u5", [CraftWorldObject.WOOD])
            automaton.add_edge("u5", "u2", [CraftWorldObject.WORKBENCH])
        else:
            automaton.add_edge("u0", "u1", [CraftWorldObject.WOOD, "~" + CraftWorldObject.WORKBENCH, "~" + CraftWorldObject.IRON])
            automaton.add_edge("u0", "u2", [CraftWorldObject.WOOD, CraftWorldObject.WORKBENCH, "~" + CraftWorldObject.TOOLSHED, CraftWorldObject.IRON])
            automaton.add_edge("u0", "u3", [CraftWorldObject.WOOD, CraftWorldObject.WORKBENCH, "~" + CraftWorldObject.IRON])
            automaton.add_edge("u0", "u4", ["~" + CraftWorldObject.WOOD, CraftWorldObject.IRON])
            automaton.add_edge("u0", "u5", [CraftWorldObject.WOOD, "~" + CraftWorldObject.WORKBENCH, CraftWorldObject.IRON])
            automaton.add_edge("u0", "u_acc", [CraftWorldObject.WOOD, CraftWorldObject.WORKBENCH, CraftWorldObject.TOOLSHED, CraftWorldObject.IRON])
            automaton.add_edge("u1", "u2", [CraftWorldObject.WORKBENCH, "~" + CraftWorldObject.TOOLSHED, CraftWorldObject.IRON])
            automaton.add_edge("u1", "u3", [CraftWorldObject.WORKBENCH, "~" + CraftWorldObject.IRON])
            automaton.add_edge("u1", "u5", ["~" + CraftWorldObject.WORKBENCH, CraftWorldObject.IRON])
            automaton.add_edge("u1", "u_acc", [CraftWorldObject.WORKBENCH, CraftWorldObject.TOOLSHED, CraftWorldObject.IRON])
            automaton.add_edge("u3", "u2", ["~" + CraftWorldObject.TOOLSHED, CraftWorldObject.IRON])
            automaton.add_edge("u3", "u_acc", [CraftWorldObject.TOOLSHED, CraftWorldObject.IRON])
            automaton.add_edge("u4", "u2", [CraftWorldObject.WOOD, CraftWorldObject.WORKBENCH, "~" + CraftWorldObject.TOOLSHED])
            automaton.add_edge("u4", "u5", [CraftWorldObject.WOOD, "~" + CraftWorldObject.WORKBENCH])
            automaton.add_edge("u4", "u_acc", [CraftWorldObject.WOOD, CraftWorldObject.WORKBENCH, CraftWorldObject.TOOLSHED])
            automaton.add_edge("u5", "u2", [CraftWorldObject.WORKBENCH, "~" + CraftWorldObject.TOOLSHED])
            automaton.add_edge("u5", "u_acc", [CraftWorldObject.WORKBENCH, CraftWorldObject.TOOLSHED])
        automaton.add_edge("u2", "u_acc", [CraftWorldObject.TOOLSHED])

        automaton.set_initial_state("u0")
        automaton.set_accept_state("u_acc")
        return automaton


class CraftWorldMakeShearsEnv(CraftWorldEnv):
    """
    Get wood, get iron, use workbench (the iron and wood can be gotten in any order).
    """
    def get_restricted_observables(self):
        return [CraftWorldObject.WOOD, CraftWorldObject.WORKBENCH, CraftWorldObject.IRON]

    def get_automaton(self):
        automaton = SubgoalAutomaton()
        automaton.add_state("u0")
        automaton.add_state("u1")
        automaton.add_state("u2")
        automaton.add_state("u3")
        automaton.add_state("u_acc")

        if self.enforce_single_observable_per_location:
            automaton.add_edge("u0", "u2", [CraftWorldObject.WOOD, "~" + CraftWorldObject.IRON])
            automaton.add_edge("u0", "u3", [CraftWorldObject.IRON])
            automaton.add_edge("u2", "u1", [CraftWorldObject.IRON])
            automaton.add_edge("u3", "u1", [CraftWorldObject.WOOD])
        else:
            automaton.add_edge("u0", "u1", [CraftWorldObject.WOOD, "~" + CraftWorldObject.WORKBENCH, CraftWorldObject.IRON])
            automaton.add_edge("u0", "u2", [CraftWorldObject.WOOD, "~" + CraftWorldObject.IRON])
            automaton.add_edge("u0", "u3", ["~" + CraftWorldObject.WOOD, CraftWorldObject.IRON])
            automaton.add_edge("u0", "u_acc", [CraftWorldObject.WOOD, CraftWorldObject.WORKBENCH, CraftWorldObject.IRON])
            automaton.add_edge("u2", "u1", ["~" + CraftWorldObject.WORKBENCH, CraftWorldObject.IRON])
            automaton.add_edge("u2", "u_acc", [CraftWorldObject.WORKBENCH, CraftWorldObject.IRON])
            automaton.add_edge("u3", "u1", [CraftWorldObject.WOOD, "~" + CraftWorldObject.WORKBENCH])
            automaton.add_edge("u3", "u_acc", [CraftWorldObject.WOOD, CraftWorldObject.WORKBENCH])
        automaton.add_edge("u1", "u_acc", [CraftWorldObject.WORKBENCH])

        automaton.set_initial_state("u0")
        automaton.set_accept_state("u_acc")
        return automaton


class CraftWorldGetGoldEnv(CraftWorldEnv):
    """
    Get iron, get wood, use factory, use bridge (the iron and wood can be gotten in any order).
    """
    def get_restricted_observables(self):
        return [CraftWorldObject.IRON, CraftWorldObject.WOOD, CraftWorldObject.FACTORY, CraftWorldObject.BRIDGE]

    def get_automaton(self):
        automaton = SubgoalAutomaton()
        automaton.add_state("u0")
        automaton.add_state("u1")
        automaton.add_state("u2")
        automaton.add_state("u3")
        automaton.add_state("u4")
        automaton.add_state("u_acc")

        if self.enforce_single_observable_per_location:
            automaton.add_edge("u0", "u2", ["~" + CraftWorldObject.WOOD, CraftWorldObject.IRON])
            automaton.add_edge("u0", "u3", [CraftWorldObject.WOOD])
            automaton.add_edge("u2", "u4", [CraftWorldObject.WOOD])
            automaton.add_edge("u3", "u4", [CraftWorldObject.IRON])
            automaton.add_edge("u4", "u1", [CraftWorldObject.FACTORY])
        else:
            automaton.add_edge("u0", "u1", [CraftWorldObject.WOOD, CraftWorldObject.FACTORY, CraftWorldObject.IRON, "~" + CraftWorldObject.BRIDGE])
            automaton.add_edge("u0", "u2", ["~" + CraftWorldObject.WOOD, CraftWorldObject.IRON])
            automaton.add_edge("u0", "u3", [CraftWorldObject.WOOD, "~" + CraftWorldObject.IRON])
            automaton.add_edge("u0", "u4", [CraftWorldObject.WOOD, "~" + CraftWorldObject.FACTORY, CraftWorldObject.IRON])
            automaton.add_edge("u0", "u_acc", [CraftWorldObject.WOOD, CraftWorldObject.FACTORY, CraftWorldObject.IRON, CraftWorldObject.BRIDGE])
            automaton.add_edge("u2", "u1", [CraftWorldObject.WOOD, CraftWorldObject.FACTORY, "~" + CraftWorldObject.BRIDGE])
            automaton.add_edge("u2", "u4", [CraftWorldObject.WOOD, "~" + CraftWorldObject.FACTORY])
            automaton.add_edge("u2", "u_acc", [CraftWorldObject.WOOD, CraftWorldObject.FACTORY, CraftWorldObject.BRIDGE])
            automaton.add_edge("u3", "u1", [CraftWorldObject.FACTORY, CraftWorldObject.IRON, "~" + CraftWorldObject.BRIDGE])
            automaton.add_edge("u3", "u4", ["~" + CraftWorldObject.FACTORY, CraftWorldObject.IRON])
            automaton.add_edge("u3", "u_acc", [CraftWorldObject.FACTORY, CraftWorldObject.IRON, CraftWorldObject.BRIDGE])
            automaton.add_edge("u4", "u1", [CraftWorldObject.FACTORY, "~" + CraftWorldObject.BRIDGE])
            automaton.add_edge("u4", "u_acc", [CraftWorldObject.FACTORY, CraftWorldObject.BRIDGE])
        automaton.add_edge("u1", "u_acc", [CraftWorldObject.BRIDGE])

        automaton.set_initial_state("u0")
        automaton.set_accept_state("u_acc")
        return automaton


class CraftWorldGetGemEnv(CraftWorldEnv):
    """
    Get wood, use workbench, get iron, use toolshed, use axe (the iron can be gotten at any time before using the toolshed).
    """
    def get_restricted_observables(self):
        return [CraftWorldObject.WOOD, CraftWorldObject.WORKBENCH, CraftWorldObject.IRON, CraftWorldObject.TOOLSHED, CraftWorldObject.AXE]

    def get_automaton(self):
        automaton = SubgoalAutomaton()
        automaton.add_state("u0")
        automaton.add_state("u1")
        automaton.add_state("u2")
        automaton.add_state("u3")
        automaton.add_state("u4")
        automaton.add_state("u5")
        automaton.add_state("u6")
        automaton.add_state("u_acc")

        if self.enforce_single_observable_per_location:
            automaton.add_edge("u0", "u1", [CraftWorldObject.WOOD, "~" + CraftWorldObject.IRON])
            automaton.add_edge("u0", "u4", [CraftWorldObject.IRON])
            automaton.add_edge("u1", "u3", [CraftWorldObject.WORKBENCH, "~" + CraftWorldObject.IRON])
            automaton.add_edge("u1", "u5", [CraftWorldObject.IRON])
            automaton.add_edge("u3", "u6", [CraftWorldObject.IRON])
            automaton.add_edge("u4", "u5", [CraftWorldObject.WOOD])
            automaton.add_edge("u5", "u6", [CraftWorldObject.WORKBENCH])
            automaton.add_edge("u6", "u2", [CraftWorldObject.TOOLSHED])
        else:
            automaton.add_edge("u0", "u1", [CraftWorldObject.WOOD, "~" + CraftWorldObject.WORKBENCH, "~" + CraftWorldObject.IRON])
            automaton.add_edge("u0", "u2", [CraftWorldObject.WOOD, CraftWorldObject.TOOLSHED, CraftWorldObject.WORKBENCH, CraftWorldObject.IRON, "~" + CraftWorldObject.AXE])
            automaton.add_edge("u0", "u3", [CraftWorldObject.WOOD, CraftWorldObject.WORKBENCH, "~" + CraftWorldObject.IRON])
            automaton.add_edge("u0", "u4", ["~" + CraftWorldObject.WOOD, CraftWorldObject.IRON])
            automaton.add_edge("u0", "u5", [CraftWorldObject.WOOD, "~" + CraftWorldObject.WORKBENCH, CraftWorldObject.IRON])
            automaton.add_edge("u0", "u6", [CraftWorldObject.WOOD, "~" + CraftWorldObject.TOOLSHED, CraftWorldObject.WORKBENCH, CraftWorldObject.IRON])
            automaton.add_edge("u0", "u_acc", [CraftWorldObject.WOOD, CraftWorldObject.TOOLSHED, CraftWorldObject.WORKBENCH, CraftWorldObject.IRON, CraftWorldObject.AXE])
            automaton.add_edge("u1", "u2", [CraftWorldObject.TOOLSHED, CraftWorldObject.WORKBENCH, CraftWorldObject.IRON, "~" + CraftWorldObject.AXE])
            automaton.add_edge("u1", "u3", [CraftWorldObject.WORKBENCH, "~" + CraftWorldObject.IRON])
            automaton.add_edge("u1", "u5", ["~" + CraftWorldObject.WORKBENCH, CraftWorldObject.IRON])
            automaton.add_edge("u1", "u6", ["~" + CraftWorldObject.TOOLSHED, CraftWorldObject.WORKBENCH, CraftWorldObject.IRON])
            automaton.add_edge("u1", "u_acc", [CraftWorldObject.TOOLSHED, CraftWorldObject.WORKBENCH, CraftWorldObject.IRON, CraftWorldObject.AXE])
            automaton.add_edge("u3", "u2", [CraftWorldObject.TOOLSHED, CraftWorldObject.IRON, "~" + CraftWorldObject.AXE])
            automaton.add_edge("u3", "u6", ["~" + CraftWorldObject.TOOLSHED, CraftWorldObject.IRON])
            automaton.add_edge("u3", "u_acc", [CraftWorldObject.TOOLSHED, CraftWorldObject.IRON, CraftWorldObject.AXE])
            automaton.add_edge("u4", "u2", [CraftWorldObject.WOOD, CraftWorldObject.TOOLSHED, CraftWorldObject.WORKBENCH, "~" + CraftWorldObject.AXE])
            automaton.add_edge("u4", "u5", [CraftWorldObject.WOOD, "~" + CraftWorldObject.WORKBENCH])
            automaton.add_edge("u4", "u6", [CraftWorldObject.WOOD, "~" + CraftWorldObject.TOOLSHED, CraftWorldObject.WORKBENCH])
            automaton.add_edge("u4", "u_acc", [CraftWorldObject.WOOD, CraftWorldObject.TOOLSHED, CraftWorldObject.WORKBENCH, CraftWorldObject.AXE])
            automaton.add_edge("u5", "u2", [CraftWorldObject.TOOLSHED, CraftWorldObject.WORKBENCH, "~" + CraftWorldObject.AXE])
            automaton.add_edge("u5", "u6", ["~" + CraftWorldObject.TOOLSHED, CraftWorldObject.WORKBENCH])
            automaton.add_edge("u5", "u_acc", [CraftWorldObject.TOOLSHED, CraftWorldObject.WORKBENCH, CraftWorldObject.AXE])
            automaton.add_edge("u6", "u2", [CraftWorldObject.TOOLSHED, "~" + CraftWorldObject.AXE])
            automaton.add_edge("u6", "u_acc", [CraftWorldObject.TOOLSHED, CraftWorldObject.AXE])
        automaton.add_edge("u2", "u_acc", [CraftWorldObject.AXE])

        automaton.set_initial_state("u0")
        automaton.set_accept_state("u_acc")
        return automaton
