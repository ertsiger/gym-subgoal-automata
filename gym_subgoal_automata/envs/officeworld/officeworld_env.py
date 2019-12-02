import random
from gym import spaces

from gym_subgoal_automata.utils.subgoal_automaton import SubgoalAutomaton
from gym_subgoal_automata.utils import utils
from gym_subgoal_automata.envs.gridworld.gridworld_env import GridWorldEnv, GridWorldActions


class OfficeWorldObject:
    AGENT = "A"
    COFFEE = "f"
    MAIL = "m"
    OFFICE = "g"
    PLANT = "n"
    ROOM_A = "a"
    ROOM_B = "b"
    ROOM_C = "c"
    ROOM_D = "d"


class OfficeWorldRoomVisits:
    VISITED_NONE = 0
    VISITED_A = 1
    VISITED_AB = 2
    VISITED_ABC = 3


class OfficeWorldEnv(GridWorldEnv):
    """
    The Office World environment
    from "Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning"
    by Rodrigo Toro Icarte, Toryn Q. Klassen, Richard Valenzano and Sheila A. McIlraith.

    Description:
    It is a grid consisting of an agent and different labelled locations:
        - f: coffee location
        - m: mail location
        - g: office location
        - n: plant location
        - a: room a
        - b: room b
        - c: room c
        - d: room d

    Observations:
    The state space is given by the position of the agent (9 * 12), whether the agent has the coffee or not (2), whether
    the agent has the mail or not (2) and the order in which the rooms have been visited (5). Thus, the number of states
    is 9 * 12 * 2 * 2 * 4 = 1728.

    Rewards:
    Different tasks (subclassed below) are defined in this environment. All of them are goal-oriented, i.e., provide
    a reward of 1 when a certain goal is achieved and 0 otherwise.
        - OfficeWorldDeliverCoffeeEnv: deliver coffee to the office while avoiding the plants.
        - OfficeWorldDeliverCoffeeAndMailEnv: deliver coffee and mail to the office while avoiding the plants.
        - OfficeWorldPatrolABCEnv: visit rooms A, B and C in that order while avoiding the plants.
        - OfficeWorldPatrolABCDEnv: visit rooms A, B, C and D in that order while avoiding the plants.
        - OfficeWorldPatrolABCDStrictEnv: visit rooms A, B, C and D in that order while avoiding the plants. The order
                                          is *strict*, e.g., when A has been visited, it cannot be revisited again.

    Actions:
    There are 4 deterministic actions:
        - 0: up
        - 1: down
        - 2: left
        - 3: right

    Acknowledgments:
    Most of the code has been reused from the original implementation by the authors of reward machines:
    https://bitbucket.org/RToroIcarte/qrm/src/master/.
    """
    MAP_GENERATION_FIELD = "generation"
    RANDOM_MAP_GENERATION_FIELD = "random"
    AVOID_ADJACENT_LOCATIONS_FIELD = "avoid_adjacent_locations"
    AVOID_AGENT_LOCATION_FIELD = "avoid_agent_location"
    AVOID_COFFEE_AND_MAIL_IN_ROOM_FIELD = "avoid_coffee_and_mail_in_room"
    NUM_PLANTS_FIELD = "num_plants"

    def __init__(self, params=None):
        super(OfficeWorldEnv, self).__init__(params)

        self.agent = None       # agent's location
        self.prev_agent = None  # previous agent location
        self.init_agent = None  # agent's initial position, for resetting

        self.locations = {}     # location of the office, a, b, c and d
        self.coffee = set()     # coffee positions
        self.mail = None        # mail position
        self.walls = set()

        # grid size
        self.height = 9
        self.width = 12

        # state
        self.has_coffee = False
        self.has_mail = False
        self.visited_rooms = OfficeWorldRoomVisits.VISITED_NONE

        # possible values for state variables
        self.num_has_coffee_values = 2
        self.num_has_mail_values = 2
        self.num_visited_room_values = 4

        self.observation_space = spaces.Discrete(self._get_num_states())

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
        if self._is_valid_position(target_pos) and (self.agent, target_pos) not in self.walls:
            self.prev_agent = self.agent
            self.agent = target_pos
            self._update_state()

        reward, is_done = self._get_reward(), self.is_terminal()
        self.is_game_over = is_done

        return self._get_state(), reward, is_done, self.get_observations()

    def _get_num_states(self):
        num_states = self.width * self.height
        if not self.hide_state_variables:
            num_states *= self.num_has_coffee_values * self.num_has_mail_values * self.num_visited_room_values
        return num_states

    def _get_state(self):
        num_states = self._get_num_states()

        state_possible_values = [self.width, self.height]
        state_variables = [self.agent[0], self.agent[1]]

        if not self.hide_state_variables:
            state_possible_values.extend([self.num_has_coffee_values, self.num_has_mail_values,
                                          self.num_visited_room_values])
            state_variables.extend([self.has_coffee, self.has_mail, self.visited_rooms])

        state_id = self.get_state_id(num_states, state_possible_values, state_variables)

        if self.use_one_hot_vector:
            return self.get_one_hot_state(num_states, state_id)

        return state_id

    def is_goal_achieved(self):
        raise NotImplementedError()

    def _get_reward(self):
        if self.is_goal_achieved():
            return 1.0
        return 0.0

    def is_terminal(self):
        if self.agent in self.locations:
            if self.locations[self.agent] == OfficeWorldObject.PLANT:
                return True
        return self.is_goal_achieved()

    def is_agent_at_office(self):
        return self.agent in self.locations and self.locations[self.agent] == OfficeWorldObject.OFFICE

    def get_observations(self):
        observations = set()

        for location in self.locations:
            if location == self.agent:
                observations.add(self.locations[location])

        if self.agent in self.coffee:
            observations.add(OfficeWorldObject.COFFEE)

        if self.agent == self.mail:
            observations.add(OfficeWorldObject.MAIL)

        return observations

    def get_observables(self):
        return [OfficeWorldObject.COFFEE, OfficeWorldObject.MAIL, OfficeWorldObject.OFFICE, OfficeWorldObject.PLANT,
                OfficeWorldObject.ROOM_A, OfficeWorldObject.ROOM_B, OfficeWorldObject.ROOM_C, OfficeWorldObject.ROOM_D]

    def get_restricted_observables(self):
        raise NotImplementedError()

    def get_automaton(self):
        raise NotImplementedError()

    def _update_state(self):
        if self.prev_agent in self.locations:
            if self.prev_agent != self.agent:
                location = self.locations[self.prev_agent]
                if self.visited_rooms == OfficeWorldRoomVisits.VISITED_NONE and location == OfficeWorldObject.ROOM_A:
                    self.visited_rooms = OfficeWorldRoomVisits.VISITED_A
                elif self.visited_rooms == OfficeWorldRoomVisits.VISITED_A and location == OfficeWorldObject.ROOM_B:
                    self.visited_rooms = OfficeWorldRoomVisits.VISITED_AB
                elif self.visited_rooms == OfficeWorldRoomVisits.VISITED_AB and location == OfficeWorldObject.ROOM_C:
                    self.visited_rooms = OfficeWorldRoomVisits.VISITED_ABC
        if self.agent in self.coffee:
            self.has_coffee = True
        if self.agent == self.mail:
            self.has_mail = True

    def reset(self):
        super().reset()

        # set initial state
        self.agent = self.init_agent
        self.prev_agent = None
        self.has_coffee = False
        self.has_mail = False
        self.visited_rooms = OfficeWorldRoomVisits.VISITED_NONE

        # update initial state according to the map layout
        self._update_state()

        return self._get_state()

    def _load_map(self):
        self._load_walls()

        generation_type = utils.get_param(self.params, OfficeWorldEnv.MAP_GENERATION_FIELD)
        if generation_type == OfficeWorldEnv.RANDOM_MAP_GENERATION_FIELD:
            self._load_random_map()
        else:
            self._load_paper_map()

    def _load_random_map(self):
        random_gen = random.Random(self.seed)

        avoid_adjacent_locations = utils.get_param(self.params, OfficeWorldEnv.AVOID_ADJACENT_LOCATIONS_FIELD, False)
        avoid_agent_location = utils.get_param(self.params, OfficeWorldEnv.AVOID_AGENT_LOCATION_FIELD, False)
        avoid_coffee_and_mail_in_room = utils.get_param(self.params, OfficeWorldEnv.AVOID_COFFEE_AND_MAIL_IN_ROOM_FIELD, False)

        # agent
        self.init_agent = (-1, -1)
        while not self._is_valid_position(self.init_agent):
            self.init_agent = (random_gen.randint(0, self.width), random_gen.randint(0, self.height))

        # mail
        self.mail = (-1, -1)
        while not self._is_valid_position(self.mail):
            self.mail = (random_gen.randint(0, self.width), random_gen.randint(0, self.height))

        # coffee
        coffee_pos = (-1, -1)
        while not self._is_valid_position(coffee_pos):
            coffee_pos = (random_gen.randint(0, self.width), random_gen.randint(0, self.height))
        self.coffee.add(coffee_pos)

        # office
        office_pos = (-1, -1)
        while not self._is_valid_position(office_pos) \
                or (avoid_coffee_and_mail_in_room and office_pos == self.mail) \
                or (avoid_coffee_and_mail_in_room and office_pos in self.coffee) \
                or (avoid_agent_location and office_pos == self.init_agent):
            office_pos = (random_gen.randint(0, self.width), random_gen.randint(0, self.height))
        self.locations[office_pos] = OfficeWorldObject.OFFICE

        # plants
        for i in range(0, utils.get_param(self.params, OfficeWorldEnv.NUM_PLANTS_FIELD, 6)):
            plant_pos = (-1, -1)
            while not self._is_valid_position(plant_pos) \
                    or plant_pos in self.locations \
                    or plant_pos == self.init_agent \
                    or plant_pos == self.mail \
                    or plant_pos in self.coffee \
                    or self._get_num_adjacent_walls(plant_pos) > 1:
                plant_pos = (random_gen.randint(0, self.width), random_gen.randint(0, self.height))
            self.locations[plant_pos] = OfficeWorldObject.PLANT

        # a, b, c and d
        for room in [OfficeWorldObject.ROOM_A, OfficeWorldObject.ROOM_B, OfficeWorldObject.ROOM_C,
                     OfficeWorldObject.ROOM_D]:
            room_pos = (-1, 1)
            while not self._is_valid_position(room_pos) \
                    or room_pos in self.locations \
                    or (avoid_coffee_and_mail_in_room and room_pos in self.coffee) \
                    or (avoid_coffee_and_mail_in_room in room_pos == self.mail) \
                    or (avoid_adjacent_locations and self._is_adjacent_to_location(room_pos)) \
                    or (avoid_agent_location and office_pos == self.init_agent):
                room_pos = (random_gen.randint(0, self.width), random_gen.randint(0, self.height))
            self.locations[room_pos] = room

    # load map as defined in the original Reward Machines paper
    def _load_paper_map(self):
        self.init_agent = (2, 1)

        self.coffee.add((8, 2))
        self.coffee.add((3, 6))
        self.mail = (7, 4)

        self.locations[(1, 1)] = OfficeWorldObject.ROOM_A
        self.locations[(10, 1)] = OfficeWorldObject.ROOM_B
        self.locations[(10, 7)] = OfficeWorldObject.ROOM_C
        self.locations[(1, 7)] = OfficeWorldObject.ROOM_D
        self.locations[(4, 4)] = OfficeWorldObject.OFFICE
        self.locations[(4, 1)] = OfficeWorldObject.PLANT
        self.locations[(7, 1)] = OfficeWorldObject.PLANT
        self.locations[(4, 7)] = OfficeWorldObject.PLANT
        self.locations[(7, 7)] = OfficeWorldObject.PLANT
        self.locations[(1, 4)] = OfficeWorldObject.PLANT
        self.locations[(10, 4)] = OfficeWorldObject.PLANT

    def _is_adjacent_to_location(self, position):
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if (x == 0 and y != 0) or (x != 0 and y == 0):
                    test_pos = (position[0] + x, position[1] + y)
                    if test_pos in self.locations:
                        return True
        return False

    def _load_walls(self):
        for x in range(2, self.width, 3):
            for y in range(0, self.height):
                if y not in [1, 7]:
                    self.walls.add(((x, y), (x + 1, y)))
                    self.walls.add(((x + 1, y), (x, y)))

        for y in range(2, self.height, 3):
            for x in range(0, self.width):
                if x not in range(1, self.width, 3):
                    self.walls.add(((x, y), (x, y + 1)))
                    self.walls.add(((x, y + 1), (x, y)))

        self.walls.add(((4, 2), (4, 3)))
        self.walls.add(((4, 3), (4, 2)))

        self.walls.add(((7, 2), (7, 3)))
        self.walls.add(((7, 3), (7, 2)))

    def _get_num_adjacent_walls(self, position):
        num_walls = 0
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if (x == 0 and y != 0) or (x != 0 and y == 0):
                    test_pos = (position[0] + x, position[1] + y)
                    if self._find_wall(test_pos):
                        num_walls += 1
        return num_walls

    def _find_wall(self, position):
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if (x == 0 and y != 0) or (x != 0 and y == 0):
                    test_pos = (position[0] + x, position[1] + y)
                    test_wall = (position, test_pos)
                    if test_wall in self.walls:
                        return True
        return False

    def _is_valid_position(self, pos):
        return 0 <= pos[0] < self.width and 0 <= pos[1] < self.height

    def render(self, mode='human'):
        self._render_horizontal_line()
        for y in range(self.height - 1, -1, -1):
            print("|", end="")
            for x in range(0, self.width):
                position = (x, y)
                if position == self.agent:
                    print(OfficeWorldObject.AGENT, end="")
                elif position in self.locations:
                    print(self.locations[position], end="")
                elif position in self.coffee:
                    print(OfficeWorldObject.COFFEE, end="")
                elif position == self.mail:
                    print(OfficeWorldObject.MAIL, end="")
                else:
                    print(" ", end="")

                wall = ((x, y), (x + 1, y))
                if wall in self.walls:
                    print("*", end="")
                else:
                    print(" ", end="")
            print("|")

            if y > 0:
                self._render_horizontal_wall(y)

        self._render_horizontal_line()

    def _render_horizontal_wall(self, y):
        print("|", end="")
        for x in range(0, self.width):
            wall = ((x, y), (x, y - 1))
            if wall in self.walls:
                print("--", end="")
            else:
                print("  ", end="")
        print("|")

    def _render_horizontal_line(self):
        for x in range(0, 2 * self.width + 2):
            print("-", end="")
        print()


class OfficeWorldDeliverCoffeeEnv(OfficeWorldEnv):
    """
    Deliver coffee to the office while avoiding the plants.
    """
    def is_goal_achieved(self):
        return self.has_coffee and self.is_agent_at_office()

    def get_restricted_observables(self):
        return [OfficeWorldObject.COFFEE, OfficeWorldObject.OFFICE, OfficeWorldObject.PLANT]

    def get_automaton(self):
        automaton = SubgoalAutomaton()
        automaton.add_state("s0")
        automaton.add_state("s1")
        automaton.add_state("s_acc")
        automaton.add_state("s_rej")
        automaton.add_edge("s0", "s1", ["f", "~g", "~n"])
        automaton.add_edge("s0", "s_acc", ["f", "g", "~n"])
        automaton.add_edge("s0", "s_rej", ["n"])
        automaton.add_edge("s1", "s_acc", ["g", "~n"])
        automaton.add_edge("s1", "s_rej", ["n"])
        automaton.set_initial_state("s0")
        automaton.set_accept_state("s_acc")
        automaton.set_reject_state("s_rej")
        return automaton


class OfficeWorldDeliverCoffeeAndMailEnv(OfficeWorldEnv):
    """
    Deliver coffee and mail to the office while avoiding the plants.
    """
    def is_goal_achieved(self):
        return self.has_coffee and self.has_mail and self.is_agent_at_office()

    def get_restricted_observables(self):
        return [OfficeWorldObject.COFFEE, OfficeWorldObject.MAIL, OfficeWorldObject.OFFICE, OfficeWorldObject.PLANT]

    def get_automaton(self):
        automaton = SubgoalAutomaton()
        automaton.add_state("s0")
        automaton.add_state("s1")
        automaton.add_state("s2")
        automaton.add_state("s3")
        automaton.add_state("s_acc")
        automaton.add_state("s_rej")
        automaton.add_edge("s0", "s1", ["f", "~g", "~m", "~n"])
        automaton.add_edge("s0", "s2", ["~f", "~g", "m", "~n"])
        automaton.add_edge("s0", "s3", ["f", "~g", "m", "~n"])
        automaton.add_edge("s0", "s_acc", ["f", "g", "m", "~n"])
        automaton.add_edge("s0", "s_rej", ["n"])
        automaton.add_edge("s1", "s3", ["~g", "m", "~n"])
        automaton.add_edge("s1", "s_acc", ["g", "m", "~n"])
        automaton.add_edge("s1", "s_rej", ["n"])
        automaton.add_edge("s2", "s3", ["~g", "f", "~n"])
        automaton.add_edge("s2", "s_acc", ["g", "f", "~n"])
        automaton.add_edge("s2", "s_rej", ["n"])
        automaton.add_edge("s3", "s_acc", ["g", "~n"])
        automaton.add_edge("s3", "s_rej", ["n"])
        automaton.set_initial_state("s0")
        automaton.set_accept_state("s_acc")
        automaton.set_reject_state("s_rej")
        return automaton


class OfficeWorldPatrolABCEnv(OfficeWorldEnv):
    """
    Visit rooms A, B and C (in that order) while avoiding the plants.
    """
    def is_goal_achieved(self):
        if self.agent in self.locations:
            return self.locations[self.agent] == OfficeWorldObject.ROOM_C and self.visited_rooms == OfficeWorldRoomVisits.VISITED_AB
        return False

    def get_restricted_observables(self):
        return [OfficeWorldObject.PLANT, OfficeWorldObject.ROOM_A, OfficeWorldObject.ROOM_B, OfficeWorldObject.ROOM_C]

    def get_automaton(self):
        automaton = SubgoalAutomaton()
        automaton.add_state("s0")
        automaton.add_state("s1")
        automaton.add_state("s2")
        automaton.add_state("s_acc")
        automaton.add_state("s_rej")
        automaton.add_edge("s0", "s1", ["a"])
        automaton.add_edge("s0", "s_rej", ["n"])
        automaton.add_edge("s1", "s2", ["b"])
        automaton.add_edge("s1", "s_rej", ["n"])
        automaton.add_edge("s2", "s_acc", ["c"])
        automaton.add_edge("s2", "s_rej", ["n"])
        automaton.set_initial_state("s0")
        automaton.set_accept_state("s_acc")
        automaton.set_reject_state("s_rej")
        return automaton


class OfficeWorldPatrolABCStrictEnv(OfficeWorldPatrolABCEnv):
    """
    Visit rooms A, B and C (in that order) while avoiding the plants. The order is *strict*. For example, B cannot be
    visited before A, and A cannot be visited after visiting B.
    """
    def is_terminal(self):
        if self.agent in self.locations:
            location = self.locations[self.agent]
            if (self.visited_rooms == OfficeWorldRoomVisits.VISITED_NONE and location in [OfficeWorldObject.ROOM_B,
                                                                                          OfficeWorldObject.ROOM_C]) or \
               (self.visited_rooms == OfficeWorldRoomVisits.VISITED_A and location in [OfficeWorldObject.ROOM_A,
                                                                                       OfficeWorldObject.ROOM_C]) or \
               (self.visited_rooms == OfficeWorldRoomVisits.VISITED_AB and location in [OfficeWorldObject.ROOM_A,
                                                                                        OfficeWorldObject.ROOM_B]):
                return True
        return super().is_terminal()

    def get_automaton(self):
        automaton = super().get_automaton()
        automaton.add_edge("s0", "s_rej", ["b"])
        automaton.add_edge("s0", "s_rej", ["c"])
        automaton.add_edge("s1", "s_rej", ["a"])
        automaton.add_edge("s1", "s_rej", ["c"])
        automaton.add_edge("s2", "s_rej", ["a"])
        automaton.add_edge("s2", "s_rej", ["b"])
        return automaton


class OfficeWorldPatrolABCDEnv(OfficeWorldEnv):
    """
    Visit rooms A, B, C and D (in that order) while avoiding the plants.
    """
    def is_goal_achieved(self):
        if self.agent in self.locations:
            return self.locations[self.agent] == OfficeWorldObject.ROOM_D and self.visited_rooms == OfficeWorldRoomVisits.VISITED_ABC
        return False

    def get_restricted_observables(self):
        return [OfficeWorldObject.PLANT, OfficeWorldObject.ROOM_A, OfficeWorldObject.ROOM_B, OfficeWorldObject.ROOM_C,
                OfficeWorldObject.ROOM_D]

    def get_automaton(self):
        automaton = SubgoalAutomaton()
        automaton.add_state("s0")
        automaton.add_state("s1")
        automaton.add_state("s2")
        automaton.add_state("s3")
        automaton.add_state("s_acc")
        automaton.add_state("s_rej")
        automaton.add_edge("s0", "s1", ["a"])
        automaton.add_edge("s0", "s_rej", ["n"])
        automaton.add_edge("s1", "s2", ["b"])
        automaton.add_edge("s1", "s_rej", ["n"])
        automaton.add_edge("s2", "s3", ["c"])
        automaton.add_edge("s2", "s_rej", ["n"])
        automaton.add_edge("s3", "s_acc", ["d"])
        automaton.add_edge("s3", "s_rej", ["n"])
        automaton.set_initial_state("s0")
        automaton.set_accept_state("s_acc")
        automaton.set_reject_state("s_rej")
        return automaton


class OfficeWorldPatrolABCDStrictEnv(OfficeWorldPatrolABCDEnv):
    """
    Visit rooms A, B, C and D (in that order) while avoiding the plants. The order is *strict*. For example, B cannot be
    visited before A, and A cannot be visited after visiting B.
    """
    def is_terminal(self):
        if self.agent in self.locations:
            location = self.locations[self.agent]
            if (self.visited_rooms == OfficeWorldRoomVisits.VISITED_NONE and location in [OfficeWorldObject.ROOM_B,
                                                                                          OfficeWorldObject.ROOM_C,
                                                                                          OfficeWorldObject.ROOM_D]) or \
               (self.visited_rooms == OfficeWorldRoomVisits.VISITED_A and location in [OfficeWorldObject.ROOM_A,
                                                                                       OfficeWorldObject.ROOM_C,
                                                                                       OfficeWorldObject.ROOM_D]) or \
               (self.visited_rooms == OfficeWorldRoomVisits.VISITED_AB and location in [OfficeWorldObject.ROOM_A,
                                                                                        OfficeWorldObject.ROOM_B,
                                                                                        OfficeWorldObject.ROOM_D]) or \
               (self.visited_rooms == OfficeWorldRoomVisits.VISITED_ABC and location in [OfficeWorldObject.ROOM_A,
                                                                                         OfficeWorldObject.ROOM_B,
                                                                                         OfficeWorldObject.ROOM_C]):
                return True
        return super().is_terminal()

    def get_automaton(self):
        automaton = super().get_automaton()
        automaton.add_edge("s0", "s_rej", ["b"])
        automaton.add_edge("s0", "s_rej", ["c"])
        automaton.add_edge("s0", "s_rej", ["d"])
        automaton.add_edge("s1", "s_rej", ["a"])
        automaton.add_edge("s1", "s_rej", ["c"])
        automaton.add_edge("s1", "s_rej", ["d"])
        automaton.add_edge("s2", "s_rej", ["a"])
        automaton.add_edge("s2", "s_rej", ["b"])
        automaton.add_edge("s2", "s_rej", ["d"])
        automaton.add_edge("s3", "s_rej", ["a"])
        automaton.add_edge("s3", "s_rej", ["b"])
        automaton.add_edge("s3", "s_rej", ["c"])
        return automaton
