import collections
import itertools
import math
import time
import gym
from gym import spaces
import numpy as np
import random
import pygame
from gym_subgoal_automata.utils import utils
from gym_subgoal_automata.utils.subgoal_automaton import SubgoalAutomaton


class WaterWorldActions:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    NONE = 4


class WaterWorldObservations:
    RED = "r"
    GREEN = "g"
    CYAN = "c"
    BLUE = "b"
    YELLOW = "y"
    MAGENTA = "m"


class Ball:
    def __init__(self, color, radius, pos, vel):
        self.color = color
        self.radius = radius
        self.pos = None
        self.vel = None
        self.update(pos, vel)

    def __str__(self):
        return "\t".join([self.color, str(self.pos[0]), str(self.pos[1]), str(self.vel[0]), str(self.vel[1])])

    def update_position(self, elapsed_time):
        self.pos += elapsed_time * self.vel

    def update(self, pos, vel):
        self.pos = np.array(pos, dtype=np.float)
        self.vel = np.array(vel, dtype=np.float)

    def is_colliding(self, ball):
        d = np.linalg.norm(self.pos - ball.pos, ord=2)
        return d <= self.radius + ball.radius


class BallAgent(Ball):
    def __init__(self, color, radius, pos, vel, vel_delta, vel_max):
        super().__init__(color, radius, pos, vel)
        self.vel_delta = float(vel_delta)
        self.vel_max = float(vel_max)

    def step(self, action):
        # updating velocity
        delta = np.array([0, 0])
        if action == WaterWorldActions.UP:
            delta = np.array([0.0, +1.0])
        elif action == WaterWorldActions.DOWN:
            delta = np.array([0.0, -1.0])
        elif action == WaterWorldActions.LEFT:
            delta = np.array([-1.0, 0.0])
        elif action == WaterWorldActions.RIGHT:
            delta = np.array([+1.0, 0.0])

        self.vel += self.vel_delta * delta

        # checking limits
        self.vel = np.clip(self.vel, -self.vel_max, self.vel_max)


BallSequence = collections.namedtuple("BallSequence", field_names=["sequence", "is_strict"])


class WaterWorldEnv(gym.Env):
    """
    The Water World environment
    from "Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning"
    by Rodrigo Toro Icarte, Toryn Q. Klassen, Richard Valenzano and Sheila A. McIlraith.

    Description:
    It consists of a 2D box containing 12 balls of different colors (2 balls per color). Each ball moves at a constant
    speed in a given direction and bounces when it collides with a wall. The agent is a white ball that can change its
    velocity in any of the four cardinal directions.

    Rewards:
    Different tasks (subclassed below) are defined in this environment. All of them are goal-oriented, i.e., provide
    a reward of 1 when a certain goal is achieved and 0 otherwise. The goal always consists of touching a sequence of
    balls in a specific order. We describe some examples:
        - WaterWorldRedGreenEnv: touch a red ball, then a green ball.
        - WaterWorldRedGreenAndMagentaYellowEnv: touch a red ball then a green ball, and touch a magenta ball then a
                                                 yellow ball. Note that the two sequences can be interleaved.
        - WaterWorldRedGreenBlueStrictEnv: touch a red ball, then a green ball. No other balls can be touched, so the
                                           agent has to avoid touching nothing but a red ball first, and nothing but a
                                           green ball afterwards.

    Actions:
    - 0: up
    - 1: down
    - 2: left
    - 3: right
    - 4: none

    Acknowledgments:
    Most of the code has been reused from the original implementation by the authors of reward machines:
    https://bitbucket.org/RToroIcarte/qrm/src/master/.
    """
    RENDERING_COLORS = {"A": (0, 0, 0),
                        WaterWorldObservations.RED: (255, 0, 0),
                        WaterWorldObservations.GREEN: (0, 255, 0),
                        WaterWorldObservations.BLUE: (0, 0, 255),
                        WaterWorldObservations.YELLOW: (255, 255, 0),
                        WaterWorldObservations.CYAN: (0, 255, 255),
                        WaterWorldObservations.MAGENTA: (255, 0, 255)
                        }
    RANDOM_SEED_FIELD = "seed"

    def __init__(self, params, sequences):
        super(WaterWorldEnv, self).__init__()

        # check input sequence
        self._check_sequences(sequences)

        # sequences of balls that have to be touched
        self.sequences = sequences
        self.state = None  # current index in each sequence

        # parameters
        self.max_x = utils.get_param(params, "max_x", 400)
        self.max_y = utils.get_param(params, "max_y", 400)
        self.ball_num_colors = len(self.get_observables())
        self.ball_radius = utils.get_param(params, "ball_radius", 15)
        self.ball_velocity = utils.get_param(params, "ball_velocity", 30)
        self.ball_num_per_color = utils.get_param(params, "ball_num_per_color", 2)
        self.use_velocities = utils.get_param(params, "use_velocities", True)
        self.ball_disappear = utils.get_param(params, "ball_disappear", False)
        self.agent_vel_delta = self.ball_velocity
        self.agent_vel_max = 3 * self.ball_velocity

        # agent ball and other balls to avoid or touch
        self.agent = None
        self.balls = []

        self.observation_space = spaces.Discrete(52)  # not exactly correct....
        self.action_space = spaces.Discrete(5)

        # whether the game is over
        self.is_game_over = False

        self.seed = utils.get_param(params, WaterWorldEnv.RANDOM_SEED_FIELD)

        # rendering attributes
        self.is_rendering = False
        self.game_display = None

        self.waiting_for_strict_observation = True

    def _check_sequences(self, sequences):
        for sequence in sequences:
            if sequence.is_strict and len(sequences) > 1:
                raise Exception("Error: Sequences containing one strict subsequence must only contain this item!")

    def _get_pos_vel_new_ball(self, random_gen):
        angle = random_gen.random() * 2 * math.pi

        if self.use_velocities:
            vel = self.ball_velocity * math.sin(angle), self.ball_velocity * math.cos(angle)
        else:
            vel = 0.0, 0.0

        while True:
            pos = 2 * self.ball_radius + random_gen.random() * (self.max_x - 2 * self.ball_radius), \
                  2 * self.ball_radius + random_gen.random() * (self.max_y - 2 * self.ball_radius)
            if not self._is_colliding(pos) and np.linalg.norm(self.agent.pos - np.array(pos), ord=2) > 4 * self.ball_radius:
                break
        return pos, vel

    def _is_colliding(self, pos):
        for b in self.balls + [self.agent]:
            if np.linalg.norm(b.pos - np.array(pos), ord=2) < 2 * self.ball_radius:
                return True
        return False

    def get_observations(self):
        return {b.color for b in self._get_current_collisions()}

    def get_observables(self):
        return [WaterWorldObservations.RED, WaterWorldObservations.GREEN, WaterWorldObservations.BLUE,
                WaterWorldObservations.CYAN, WaterWorldObservations.MAGENTA, WaterWorldObservations.YELLOW]

    def get_restricted_observables(self):
        return self._get_symbols_from_sequence()

    def _get_current_collisions(self):
        collisions = set()
        for b in self.balls:
            if self.agent.is_colliding(b):
                collisions.add(b)
        return collisions

    def is_terminal(self):
        return self.is_game_over

    def step(self, action, elapsed_time=0.1):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if self.is_game_over:
            return self._get_features(), 0.0, True, self.get_observations()

        # if balls disappear, then relocate balls that the agent is colliding before the action
        if self.ball_disappear:
            for b in self.balls:
                if self.agent.is_colliding(b):
                    pos, vel = self._get_pos_vel_new_ball()
                    b.update(pos, vel)

        # updating the agents velocity
        self.agent.step(action)
        balls_all = [self.agent] + self.balls
        max_x, max_y = self.max_x, self.max_y

        # updating position
        for b in balls_all:
            b.update_position(elapsed_time)

        # handling collisions
        for i in range(len(balls_all)):
            b = balls_all[i]
            # walls
            if b.pos[0] - b.radius < 0 or b.pos[0] + b.radius > max_x:
                # place ball against edge
                if b.pos[0] - b.radius < 0:
                    b.pos[0] = b.radius
                else:
                    b.pos[0] = max_x - b.radius
                # reverse direction
                b.vel *= np.array([-1.0, 1.0])
            if b.pos[1] - b.radius < 0 or b.pos[1] + b.radius > max_y:
                # place ball against edge
                if b.pos[1] - b.radius < 0:
                    b.pos[1] = b.radius
                else:
                    b.pos[1] = max_y - b.radius
                # reverse direction
                b.vel *= np.array([1.0, -1.0])

        observations = self.get_observations()
        reward, is_done = self._step(observations)

        if is_done:
            self.is_game_over = True

        return self._get_features(), reward, is_done, observations

    def _step(self, observations):
        reached_terminal_state = self._update_state(observations)
        if reached_terminal_state:
            return 0.0, True

        if self.is_goal_achieved():
            return 1.0, True

        return 0.0, False

    def _update_state(self, observations):
        for i in range(0, len(self.sequences)):
            current_index = self.state[i]
            sequence = self.sequences[i]
            if sequence.is_strict:
                if len(observations) == 0:
                    self.waiting_for_strict_observation = True
                elif self.waiting_for_strict_observation and len(observations) > 0 and current_index < len(sequence.sequence):
                    if len(observations) == 1 and sequence.sequence[current_index] in observations:
                        self.state[i] = current_index + 1
                        self.waiting_for_strict_observation = False
                    else:
                        return True
            else:
                while current_index < len(sequence.sequence) and sequence.sequence[current_index] in observations:
                    current_index += 1
                self.state[i] = current_index

        return False

    def is_goal_achieved(self):
        # all sequences have been observed
        for i in range(0, len(self.sequences)):
            sequence = self.sequences[i].sequence
            if self.state[i] != len(sequence):
                return False
        return True

    def _get_features(self):
        # absolute position and velocity of the agent + relative positions and velocities of the other balls with
        # respect to the agent
        agent, balls = self.agent, self.balls

        if self.use_velocities:
            n_features = 4 + len(balls) * 4
            features = np.zeros(n_features, dtype=np.float32)

            pos_max = np.array([float(self.max_x), float(self.max_y)])
            vel_max = float(self.ball_velocity + self.agent_vel_max)

            features[0:2] = agent.pos / pos_max
            features[2:4] = agent.vel / float(self.agent_vel_max)

            for i in range(len(balls)):
                # if the balls are colliding, they are not included because there is nothing the agent can do about it
                b = balls[i]
                if not self.ball_disappear or not agent.is_colliding(b):
                    init = 4 * (i + 1)
                    features[init:init+2] = (b.pos - agent.pos) / pos_max
                    features[init+2:init+4] = (b.vel - agent.vel) / vel_max
        else:
            n_features = 4 + len(balls) * 2
            features = np.zeros(n_features, dtype=np.float)

            pos_max = np.array([float(self.max_x), float(self.max_y)])

            features[0:2] = agent.pos / pos_max
            features[2:4] = agent.vel / float(self.agent_vel_max)

            for i in range(len(balls)):
                b = balls[i]
                if not self.ball_disappear or not agent.is_colliding(b):
                    init = 2 * i + 4
                    features[init:init+2] = (b.pos - agent.pos) / pos_max

        return features

    def reset(self):
        self.is_game_over = False
        self.waiting_for_strict_observation = True

        random_gen = random.Random(self.seed)

        # adding the agent
        pos_a = [2 * self.ball_radius + random_gen.random() * (self.max_x - 2 * self.ball_radius),
                 2 * self.ball_radius + random_gen.random() * (self.max_y - 2 * self.ball_radius)]
        self.agent = BallAgent("A", self.ball_radius, pos_a, [0.0, 0.0], self.agent_vel_delta, self.agent_vel_max)

        # adding the balls
        self.balls = []
        colors = self.get_observables()
        for c in range(self.ball_num_colors):
            for _ in range(self.ball_num_per_color):
                color = colors[c]
                pos, vel = self._get_pos_vel_new_ball(random_gen)
                ball = Ball(color, self.ball_radius, pos, vel)
                self.balls.append(ball)

        # reset current index in each sequence
        self.state = [0] * len(self.sequences)

        return self._get_features()

    def render(self, mode='human'):
        if not self.is_rendering:
            pygame.init()
            pygame.display.set_caption("Water World")
            self.game_display = pygame.display.set_mode((self.max_x, self.max_y))
            self.is_rendering = True

        # printing image
        self.game_display.fill((255, 255, 255))
        for ball in self.balls:
            self._render_ball(self.game_display, ball, 0)
        self._render_ball(self.game_display, self.agent, 3)

        pygame.display.update()

    def _render_ball(self, game_display, ball, thickness):
        pygame.draw.circle(game_display, WaterWorldEnv.RENDERING_COLORS[ball.color],
                           self._get_ball_position(ball, self.max_y), ball.radius, thickness)

    def _get_ball_position(self, ball, max_y):
        return int(round(ball.pos[0])), int(max_y) - int(round(ball.pos[1]))

    def close(self):
        pygame.quit()
        self.is_rendering = False

    def get_automaton(self):
        automaton = SubgoalAutomaton()

        if self._is_strict_sequence():
            self._add_strict_transitions_to_automaton(automaton)
            automaton.set_reject_state("s_rej")
        else:
            self._add_transitions_to_automaton(automaton)

        automaton.set_initial_state("s0")
        automaton.set_accept_state("s_acc")

        return automaton

    def _is_strict_sequence(self):
        for sequence in self.sequences:
            if sequence.is_strict:
                return True
        return False

    def _add_strict_transitions_to_automaton(self, automaton):
        sequence = self.sequences[0].sequence
        state_counter = 0

        for i in range(len(sequence)):
            symbol = sequence[i]
            current_state = "s%d" % state_counter
            if i == len(sequence) - 1:
                next_state = "s_acc"
            else:
                next_state = "s%d" % (state_counter + 1)
            automaton.add_edge(current_state, next_state, [symbol])
            for other_symbol in self.get_observables():
                if symbol != other_symbol:
                    automaton.add_edge(current_state, "s_rej", [other_symbol, "~" + symbol])
            state_counter += 1

    def _add_transitions_to_automaton(self, automaton):
        symbols = self._get_symbols_from_sequence()
        seq_tuple = self._get_sequence_tuple_from_ball_sequence()

        seq_tuples_to_states = {}
        current_state_id = 0

        queue = [seq_tuple]
        checked_tuples = set()

        while len(queue) > 0:
            seq_tuple = queue.pop(0)
            checked_tuples.add(seq_tuple)

            derived_to_transitions = {}

            # test all possible assignments of symbols to the current sequence and derive new sequences, which
            # correspond to all the possible children in the automaton for that state
            for l in range(len(symbols) + 1):
                for subset in itertools.combinations(symbols, l):
                    derived_tuple = self._get_derived_sequence_tuple_from_assignment(seq_tuple, subset)
                    if seq_tuple != derived_tuple:  # if the derivation is the same than the original sequence, discard it
                        if derived_tuple not in derived_to_transitions:
                            derived_to_transitions[derived_tuple] = []
                        derived_to_transitions[derived_tuple].append(subset)

                    # each tuple corresponds to a specific state
                    if derived_tuple not in seq_tuples_to_states:
                        if len(derived_tuple) == 0:
                            seq_tuples_to_states[derived_tuple] = "s_acc"
                        else:
                            seq_tuples_to_states[derived_tuple] = "s%d" % current_state_id
                            current_state_id += 1

                    # append the derived sequence to the queue to be analysed
                    if derived_tuple not in checked_tuples and derived_tuple not in queue:
                        queue.append(derived_tuple)

            # compress all the possible transitions to a derived sequence by checking which symbols never change their
            # value in these transitions (i.e., always appear as true or always appear as false)
            for derived_seq in derived_to_transitions:
                if derived_seq not in checked_tuples:
                    final_transition = []
                    for symbol in symbols:
                        if self._is_symbol_false_in_all_arrays(symbol, derived_to_transitions[derived_seq]):
                            final_transition.append("~" + symbol)
                        elif self._is_symbol_true_in_all_arrays(symbol, derived_to_transitions[derived_seq]):
                            final_transition.append(symbol)
                    to_state = seq_tuples_to_states[derived_seq]
                    automaton.add_edge(seq_tuples_to_states[seq_tuple], to_state, final_transition)

    def _get_symbols_from_sequence(self):
        symbols = set([])
        for sequence in self.sequences:
            for symbol in sequence.sequence:
                symbols.add(symbol)
        return sorted(list(symbols))

    def _get_sequence_tuple_from_ball_sequence(self):
        tuple_seq = []
        for sequence in self.sequences:
            tuple_seq.append(tuple(sequence.sequence))
        return tuple(tuple_seq)

    def _get_derived_sequence_tuple_from_assignment(self, seq_tuple, assignment):
        derived_tuple = []

        # apply the assignment to each subsequence until it no longer holds
        for subsequence in seq_tuple:
            last_unsat = 0
            for i in range(len(subsequence)):
                if subsequence[i] in assignment:
                    last_unsat += 1
                else:
                    break
            if last_unsat < len(subsequence):
                derived_tuple.append(subsequence[last_unsat:])

        return tuple(derived_tuple)

    def _is_symbol_false_in_all_arrays(self, symbol, arrays):
        for array in arrays:
            if symbol in array:
                return False
        return True

    def _is_symbol_true_in_all_arrays(self, symbol, arrays):
        for array in arrays:
            if symbol not in array:
                return False
        return True

    def play(self):
        self.reset()
        self.render()

        clock = pygame.time.Clock()

        t_previous = time.time()
        actions = set()

        total_reward = 0.0

        while not self.is_terminal():
            for event in pygame.event.get():
                if event.type == pygame.KEYUP:
                    if WaterWorldActions.LEFT in actions and event.key == pygame.K_LEFT:
                        actions.remove(WaterWorldActions.LEFT)
                    elif WaterWorldActions.RIGHT in actions and event.key == pygame.K_RIGHT:
                        actions.remove(WaterWorldActions.RIGHT)
                    elif WaterWorldActions.UP in actions and event.key == pygame.K_UP:
                        actions.remove(WaterWorldActions.UP)
                    elif WaterWorldActions.DOWN in actions and event.key == pygame.K_DOWN:
                        actions.remove(WaterWorldActions.DOWN)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        actions.add(WaterWorldActions.LEFT)
                    elif event.key == pygame.K_RIGHT:
                        actions.add(WaterWorldActions.RIGHT)
                    elif event.key == pygame.K_UP:
                        actions.add(WaterWorldActions.UP)
                    elif event.key == pygame.K_DOWN:
                        actions.add(WaterWorldActions.DOWN)

            t_current = time.time()
            t_delta = (t_current - t_previous)

            # getting the action
            if len(actions) == 0:
                a = WaterWorldActions.NONE
            else:
                a = random.choice(list(actions))

            # executing the action
            _, reward, is_done, _ = self.step(a, t_delta)
            total_reward += reward

            # printing image
            self.render()

            clock.tick(20)

            t_previous = t_current

        print("Game finished. Total reward: %.2f." % total_reward)

        self.close()


class WaterWorldRedGreenEnv(WaterWorldEnv):
    def __init__(self, params=None):
        sequences = [BallSequence([WaterWorldObservations.RED, WaterWorldObservations.GREEN], False)]
        super(WaterWorldRedGreenEnv, self).__init__(params, sequences)


class WaterWorldBlueCyanEnv(WaterWorldEnv):
    def __init__(self, params=None):
        sequences = [BallSequence([WaterWorldObservations.BLUE, WaterWorldObservations.CYAN], False)]
        super(WaterWorldBlueCyanEnv, self).__init__(params, sequences)


class WaterWorldMagentaYellowEnv(WaterWorldEnv):
    def __init__(self, params=None):
        sequences = [BallSequence([WaterWorldObservations.MAGENTA, WaterWorldObservations.YELLOW], False)]
        super(WaterWorldMagentaYellowEnv, self).__init__(params, sequences)


class WaterWorldRedGreenAndBlueCyanEnv(WaterWorldEnv):
    def __init__(self, params=None):
        sequences = [BallSequence([WaterWorldObservations.RED, WaterWorldObservations.GREEN], False),
                     BallSequence([WaterWorldObservations.BLUE, WaterWorldObservations.CYAN], False)]
        super(WaterWorldRedGreenAndBlueCyanEnv, self).__init__(params, sequences)


class WaterWorldBlueCyanAndMagentaYellowEnv(WaterWorldEnv):
    def __init__(self, params=None):
        sequences = [BallSequence([WaterWorldObservations.BLUE, WaterWorldObservations.CYAN], False),
                     BallSequence([WaterWorldObservations.MAGENTA, WaterWorldObservations.YELLOW], False)]
        super(WaterWorldBlueCyanAndMagentaYellowEnv, self).__init__(params, sequences)


class WaterWorldRedGreenAndMagentaYellowEnv(WaterWorldEnv):
    def __init__(self, params=None):
        sequences = [BallSequence([WaterWorldObservations.RED, WaterWorldObservations.GREEN], False),
                     BallSequence([WaterWorldObservations.MAGENTA, WaterWorldObservations.YELLOW], False)]
        super(WaterWorldRedGreenAndMagentaYellowEnv, self).__init__(params, sequences)


class WaterWorldRedGreenAndBlueCyanAndMagentaYellowEnv(WaterWorldEnv):
    def __init__(self, params=None):
        sequences = [BallSequence([WaterWorldObservations.RED, WaterWorldObservations.GREEN], False),
                     BallSequence([WaterWorldObservations.BLUE, WaterWorldObservations.CYAN], False),
                     BallSequence([WaterWorldObservations.MAGENTA, WaterWorldObservations.YELLOW], False)]
        super(WaterWorldRedGreenAndBlueCyanAndMagentaYellowEnv, self).__init__(params, sequences)


class WaterWorldRedGreenBlueAndCyanMagentaYellowEnv(WaterWorldEnv):
    def __init__(self, params=None):
        sequences = [BallSequence([WaterWorldObservations.RED,
                                   WaterWorldObservations.GREEN,
                                   WaterWorldObservations.BLUE], False),
                     BallSequence([WaterWorldObservations.CYAN,
                                   WaterWorldObservations.MAGENTA,
                                   WaterWorldObservations.YELLOW], False)]
        super(WaterWorldRedGreenBlueAndCyanMagentaYellowEnv, self).__init__(params, sequences)


class WaterWorldRedGreenBlueStrictEnv(WaterWorldEnv):
    def __init__(self, params=None):
        sequences = [BallSequence([WaterWorldObservations.RED,
                                   WaterWorldObservations.GREEN,
                                   WaterWorldObservations.BLUE], True)]
        super(WaterWorldRedGreenBlueStrictEnv, self).__init__(params, sequences)


class WaterWorldCyanMagentaYellowStrictEnv(WaterWorldEnv):
    def __init__(self, params=None):
        sequences = [BallSequence([WaterWorldObservations.CYAN,
                                   WaterWorldObservations.MAGENTA,
                                   WaterWorldObservations.YELLOW], True)]
        super(WaterWorldCyanMagentaYellowStrictEnv, self).__init__(params, sequences)


class WaterWorldRedAndBlueCyanEnv(WaterWorldEnv):
    def __init__(self, params=None):
        sequences = [BallSequence([WaterWorldObservations.RED], False),
                     BallSequence([WaterWorldObservations.BLUE,
                                   WaterWorldObservations.CYAN], False)]
        super(WaterWorldRedAndBlueCyanEnv, self).__init__(params, sequences)


class WaterWorldRedGreenAndBlueEnv(WaterWorldEnv):
    def __init__(self, params=None):
        sequences = [BallSequence([WaterWorldObservations.RED,
                                   WaterWorldObservations.GREEN], False),
                     BallSequence([WaterWorldObservations.BLUE], False)]
        super(WaterWorldRedGreenAndBlueEnv, self).__init__(params, sequences)


class WaterWorldRedGreenBlueEnv(WaterWorldEnv):
    def __init__(self, params=None):
        sequences = [BallSequence([WaterWorldObservations.RED,
                                   WaterWorldObservations.GREEN,
                                   WaterWorldObservations.BLUE], False)]
        super(WaterWorldRedGreenBlueEnv, self).__init__(params, sequences)


class WaterWorldRedGreenBlueCyanEnv(WaterWorldEnv):
    def __init__(self, params=None):
        sequences = [BallSequence([WaterWorldObservations.RED,
                                   WaterWorldObservations.GREEN,
                                   WaterWorldObservations.BLUE,
                                   WaterWorldObservations.CYAN], False)]
        super(WaterWorldRedGreenBlueCyanEnv, self).__init__(params, sequences)

