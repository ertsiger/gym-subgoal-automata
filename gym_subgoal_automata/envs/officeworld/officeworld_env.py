from abc import abstractmethod
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
    VISITED_POSSIBLE_VALUES = 4


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
        - OfficeWorldDeliverMailEnv: deliver mail to the office while avoiding the plants.
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

    Constraints:
    The map has the following constraints:
        - The agent cannot be placed together with A-D and plants.
        - Objects A, B, C, D and the office cannot be in the same location.
        - The plants do not share a location with any other object.
        - A coffee location and the mail can be in the same location. For random map generation, the flag
          "allow_share_location_coffee_mail" must be true (it is true by default).
        - The coffee and the mail can share their location with A-D and the office. For random map generation, the flag
          "allow_coffee_mail_in_room" must be true (it is true by default).

    Map generation modes:
    There are three ways to generate the map (use the "generation" parameter below):
        - "random": puts the objects randomly in the grid. It makes sure that (1) plants and A-D do not obstruct corridors
                    between rooms, (2) plants are not adjacent to other plants or A-D unless there is a wall between them,
                    and (3) a location A-D is not adjacent to another location A-D unless there is a wall between them.
                    These constraints on plants and A-D are imposed so that randomly generated grids can be solved.
        - "params": the locations of the objects are given through the "map" parameter.
        - "paper": the configuration of the map used in the paper above is used.

    Acknowledgments:
    Most of the code has been reused from the original implementation by the authors of reward machines:
    https://bitbucket.org/RToroIcarte/qrm/src/master/.
    """
    MAP_GENERATION_FIELD = "generation"  # how to assign the objects to different tiles in the map

    RANDOM_MAP_GENERATION = "random"                                       # assign objects randomly
    NUM_PLANTS_FIELD = "num_plants"                                        # how many plants/decorations to generate
    NUM_COFFEES_FIELD = "num_coffees"                                      # how many coffee locations to generate
    ALLOW_SHARE_LOCATION_COFFEE_MAIL = "allow_share_location_coffee_mail"  # the coffee and the mail can share location
    ALLOW_COFFEE_MAIL_IN_ROOM = "allow_coffee_mail_in_room"                # allow to generate coffee and/or mail inside
                                                                           # a room (a-d, office)

    PARAMS_MAP_GENERATION = "params"  # use the object assignment specified in the MAP_PARAM parameter in 'params'
    MAP_PARAM = "map"

    PAPER_MAP_GENERATION = "paper"  # use the object assignment given in the reward machines paper

    DROP_COFFEE_ON_PLANT_ENABLE = "drop_coffee_enable"  # whether the agent can drop the coffee when it steps on a plant

    def __init__(self, params=None):
        super().__init__(params)

        self.agent = None       # agent's location
        self.prev_agent = None  # previous agent location
        self.init_agent = None  # agent's initial position, for resetting

        self.locations = {}     # location of the office, a, b, c and d
        self.coffee = set()     # coffee positions
        self.mail = None        # mail position
        self.walls = set()      # pairs of locations between which there is a wall
        self.corridor_locations = {(2, 1), (3, 1), (5, 1), (6, 1), (8, 1), (9, 1),
                                   (2, 7), (3, 7), (5, 7), (6, 7), (8, 7), (9, 7),
                                   (1, 2), (1, 3), (1, 5), (1, 6),
                                   (4, 5), (4, 6), (7, 5), (7, 6),
                                   (10, 2), (10, 3), (10, 5), (10, 6)}

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
        self.num_visited_room_values = OfficeWorldRoomVisits.VISITED_POSSIBLE_VALUES

        self.observation_space = spaces.Discrete(self._get_num_states())

        # random generation parameters
        self.allow_coffee_and_mail_in_room = utils.get_param(self.params, OfficeWorldEnv.ALLOW_COFFEE_MAIL_IN_ROOM, True)
        self.can_share_location_coffee_mail = utils.get_param(self.params, OfficeWorldEnv.ALLOW_SHARE_LOCATION_COFFEE_MAIL, True)

        self.drop_coffee_on_plant_enable = utils.get_param(self.params, OfficeWorldEnv.DROP_COFFEE_ON_PLANT_ENABLE, False)

        if params is not None:
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
        if self._is_valid_movement(self.agent, target_pos):
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

    @abstractmethod
    def is_goal_achieved(self):
        pass

    def _get_reward(self):
        return 1.0 if self.is_goal_achieved() else 0.0

    def is_terminal(self):
        return self.is_deadend() or self.is_goal_achieved()

    def is_deadend(self):
        return self.is_agent_on_plant() and not self.drop_coffee_on_plant_enable

    def is_agent_at_office(self):
        return self.agent in self.locations and self.locations[self.agent] == OfficeWorldObject.OFFICE

    def is_agent_on_plant(self):
        return self.agent in self.locations and self.locations[self.agent] == OfficeWorldObject.PLANT

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
        if self.drop_coffee_on_plant_enable and self.is_agent_on_plant() and self.has_coffee:
            self.has_coffee = False

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

    """
    Map loading methods
    """
    def _load_map(self):
        self._load_walls()

        generation_type = utils.get_param(self.params, OfficeWorldEnv.MAP_GENERATION_FIELD)
        if generation_type == OfficeWorldEnv.RANDOM_MAP_GENERATION:
            self._load_random_map()
        elif generation_type == OfficeWorldEnv.PARAMS_MAP_GENERATION:
            self._load_map_from_params()
        elif generation_type == OfficeWorldEnv.PAPER_MAP_GENERATION:
            self._load_paper_map()
        else:
            raise RuntimeError("Error: Unknown map generation mode '{}'.".format(generation_type))

    def _load_random_map(self):
        random_gen = random.Random(self.seed)

        # agent
        self.init_agent = self._generate_random_position(random_gen)

        # office
        self.locations[self._generate_office_random_position(random_gen)] = OfficeWorldObject.OFFICE

        # locations a, b, c and d
        for r in [OfficeWorldObject.ROOM_A, OfficeWorldObject.ROOM_B, OfficeWorldObject.ROOM_C, OfficeWorldObject.ROOM_D]:
            room_pos = self._generate_room_random_position(random_gen)
            self.locations[room_pos] = r

        # coffee
        for i in range(utils.get_param(self.params, OfficeWorldEnv.NUM_COFFEES_FIELD, 2)):
            coffee_pos = self._generate_coffee_random_position(random_gen)
            self.coffee.add(coffee_pos)

        # mail
        self.mail = self._generate_mail_random_position(random_gen)

        # plants (make sure it is done the last to avoid clashes with other objects)
        for i in range(utils.get_param(self.params, OfficeWorldEnv.NUM_PLANTS_FIELD, 6)):
            plant_pos = self._generate_plant_random_position(random_gen)
            self.locations[plant_pos] = OfficeWorldObject.PLANT

    def _generate_random_position(self, random_gen):
        return random_gen.randint(0, self.width - 1), random_gen.randint(0, self.height - 1)

    def _generate_office_random_position(self, random_gen):
        office_pos = (-1, -1)
        while not self._is_valid_position(office_pos) \
                or office_pos in self.locations:
            office_pos = self._generate_random_position(random_gen)
        return office_pos

    def _generate_room_random_position(self, random_gen):
        room_pos = (-1, -1)
        while not self._is_valid_position(room_pos) \
                or room_pos == self.init_agent \
                or room_pos in self.locations \
                or room_pos in self.corridor_locations \
                or self._get_num_adjacent_rooms(room_pos) > 0 \
                or self._get_num_adjacent_plants(room_pos) > 0:
            room_pos = self._generate_random_position(random_gen)
        return room_pos

    def _generate_coffee_random_position(self, random_gen):
        item_pos = (-1, -1)
        while not self._is_valid_position(item_pos) \
                or item_pos in self.coffee \
                or (not self.can_share_location_coffee_mail and item_pos == self.mail) \
                or (not self.allow_coffee_and_mail_in_room and item_pos in self.locations):
            item_pos = self._generate_random_position(random_gen)
        return item_pos

    def _generate_mail_random_position(self, random_gen):
        item_pos = (-1, -1)
        while not self._is_valid_position(item_pos) \
                or (not self.can_share_location_coffee_mail and item_pos in self.coffee) \
                or (not self.allow_coffee_and_mail_in_room and item_pos in self.locations):
            item_pos = self._generate_random_position(random_gen)
        return item_pos

    def _generate_plant_random_position(self, random_gen):
        plant_pos = (-1, -1)
        while not self._is_valid_position(plant_pos) \
                or plant_pos in self.locations \
                or plant_pos == self.init_agent \
                or plant_pos == self.mail \
                or plant_pos in self.coffee \
                or plant_pos in self.corridor_locations \
                or self._get_num_adjacent_plants(plant_pos) > 0 \
                or self._get_num_adjacent_rooms(plant_pos) > 0:
            plant_pos = self._generate_random_position(random_gen)
        return plant_pos

    def _load_map_from_params(self):
        map = self.params[OfficeWorldEnv.MAP_PARAM]

        self.init_agent = map[OfficeWorldObject.AGENT]
        for location in map[OfficeWorldObject.COFFEE]:
            self.coffee.add(location)

        self.mail = map[OfficeWorldObject.MAIL]

        for obj in [OfficeWorldObject.ROOM_A, OfficeWorldObject.ROOM_B, OfficeWorldObject.ROOM_C,
                    OfficeWorldObject.ROOM_D, OfficeWorldObject.OFFICE]:
            self.locations[map[obj]] = obj

        for location in map[OfficeWorldObject.PLANT]:
            self.locations[location] = OfficeWorldObject.PLANT

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

    def _load_walls(self):
        for x in range(2, self.width - 1, 3):
            for y in range(0, self.height):
                if y not in [1, 7]:
                    self.walls.add(((x, y), (x + 1, y)))
                    self.walls.add(((x + 1, y), (x, y)))

        for y in range(2, self.height - 1, 3):
            for x in range(0, self.width):
                if x not in range(1, self.width, 3):
                    self.walls.add(((x, y), (x, y + 1)))
                    self.walls.add(((x, y + 1), (x, y)))

        self.walls.add(((4, 2), (4, 3)))
        self.walls.add(((4, 3), (4, 2)))

        self.walls.add(((7, 2), (7, 3)))
        self.walls.add(((7, 3), (7, 2)))

    def _get_num_adjacent_plants(self, position):
        """
        Returns the number of plants around a given position (i.e., in the 8 surrounding tiles). If there is a wall
        between the tiles, the plant is not counted.
        """
        return self._get_num_adjacent_locations(position, [OfficeWorldObject.PLANT])

    def _get_num_adjacent_rooms(self, position):
        """
        Returns the number of rooms (A-D) around a given position (i.e., in the 8 surrounding tiles). If there is a wall
        between the tiles, the room is not counted.
        """
        return self._get_num_adjacent_locations(position, [OfficeWorldObject.ROOM_A, OfficeWorldObject.ROOM_B,
                                                           OfficeWorldObject.ROOM_C, OfficeWorldObject.ROOM_D])

    def _get_num_adjacent_locations(self, position, location_classes):
        num_adjacent = 0
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                test_pos = (position[0] + x, position[1] + y)
                test_wall = (position, test_pos)
                if (x, y) != (0, 0) \
                        and test_wall not in self.walls \
                        and self._is_valid_position(test_pos) \
                        and test_pos in self.locations \
                        and self.locations[test_pos] in location_classes:
                    num_adjacent += 1
        return num_adjacent

    def _is_valid_position(self, pos):
        return 0 <= pos[0] < self.width and 0 <= pos[1] < self.height

    def _is_valid_movement(self, src_pos, dst_pos):
        return self._is_valid_position(dst_pos) and (src_pos, dst_pos) not in self.walls

    """
    Grid rendering
    """
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
        automaton.add_state("u0")
        automaton.add_state("u1")
        automaton.add_state("u_acc")

        automaton.add_edge("u0", "u1", [OfficeWorldObject.COFFEE, "~" + OfficeWorldObject.OFFICE])
        automaton.add_edge("u0", "u_acc", [OfficeWorldObject.COFFEE, OfficeWorldObject.OFFICE])
        automaton.add_edge("u1", "u_acc", [OfficeWorldObject.OFFICE, "~" + OfficeWorldObject.PLANT])

        if self.drop_coffee_on_plant_enable:
            automaton.add_edge("u1", "u0", [OfficeWorldObject.PLANT])
        else:
            automaton.add_state("u_rej")
            automaton.set_reject_state("u_rej")
            automaton.add_edge("u0", "u_rej", [OfficeWorldObject.PLANT, "~" + OfficeWorldObject.COFFEE, "~" + OfficeWorldObject.OFFICE])
            automaton.add_edge("u1", "u_rej", [OfficeWorldObject.PLANT])

        automaton.set_initial_state("u0")
        automaton.set_accept_state("u_acc")
        return automaton


class OfficeWorldDeliverMailEnv(OfficeWorldEnv):
    """
    Deliver coffee to the office while avoiding the plants.
    """
    def is_goal_achieved(self):
        return self.has_mail and self.is_agent_at_office()

    def get_restricted_observables(self):
        return [OfficeWorldObject.MAIL, OfficeWorldObject.OFFICE, OfficeWorldObject.PLANT]

    def get_automaton(self):
        automaton = SubgoalAutomaton()
        automaton.add_state("u0")
        automaton.add_state("u1")
        automaton.add_state("u_acc")
        automaton.add_state("u_rej")
        automaton.add_edge("u0", "u1", [OfficeWorldObject.MAIL, "~" + OfficeWorldObject.OFFICE, "~" + OfficeWorldObject.PLANT])
        automaton.add_edge("u0", "u_acc", [OfficeWorldObject.MAIL, OfficeWorldObject.OFFICE, "~" + OfficeWorldObject.PLANT])
        automaton.add_edge("u0", "u_rej", [OfficeWorldObject.PLANT])
        automaton.add_edge("u1", "u_acc", [OfficeWorldObject.OFFICE, "~" + OfficeWorldObject.PLANT])
        automaton.add_edge("u1", "u_rej", [OfficeWorldObject.PLANT])
        automaton.set_initial_state("u0")
        automaton.set_accept_state("u_acc")
        automaton.set_reject_state("u_rej")
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
        automaton.add_state("u0")
        automaton.add_state("u1")
        automaton.add_state("u2")
        automaton.add_state("u3")
        automaton.add_state("u_acc")

        automaton.add_edge("u0", "u1", [OfficeWorldObject.COFFEE, "~" + OfficeWorldObject.OFFICE, "~" + OfficeWorldObject.MAIL])
        automaton.add_edge("u0", "u2", ["~" + OfficeWorldObject.COFFEE, "~" + OfficeWorldObject.OFFICE, OfficeWorldObject.MAIL])
        automaton.add_edge("u0", "u3", [OfficeWorldObject.COFFEE, "~" + OfficeWorldObject.OFFICE, OfficeWorldObject.MAIL])
        automaton.add_edge("u0", "u_acc", [OfficeWorldObject.COFFEE, OfficeWorldObject.OFFICE, OfficeWorldObject.MAIL])
        automaton.add_edge("u1", "u3", ["~" + OfficeWorldObject.OFFICE, OfficeWorldObject.MAIL, "~" + OfficeWorldObject.PLANT])
        automaton.add_edge("u1", "u_acc", [OfficeWorldObject.OFFICE, OfficeWorldObject.MAIL, "~" + OfficeWorldObject.PLANT])
        automaton.add_edge("u2", "u3", ["~" + OfficeWorldObject.OFFICE, OfficeWorldObject.COFFEE])
        automaton.add_edge("u2", "u_acc", [OfficeWorldObject.OFFICE, OfficeWorldObject.COFFEE])
        automaton.add_edge("u3", "u_acc", [OfficeWorldObject.OFFICE, "~" + OfficeWorldObject.PLANT])

        if self.drop_coffee_on_plant_enable:
            automaton.add_edge("u1", "u0", [OfficeWorldObject.PLANT])
            automaton.add_edge("u3", "u2", [OfficeWorldObject.PLANT])
        else:
            automaton.add_state("u_rej")
            automaton.set_reject_state("u_rej")
            automaton.add_edge("u0", "u_rej", [OfficeWorldObject.PLANT, "~" + OfficeWorldObject.COFFEE, "~" + OfficeWorldObject.MAIL])
            automaton.add_edge("u1", "u_rej", [OfficeWorldObject.PLANT])
            automaton.add_edge("u2", "u_rej", [OfficeWorldObject.PLANT, "~" + OfficeWorldObject.COFFEE])
            automaton.add_edge("u3", "u_rej", [OfficeWorldObject.PLANT])

        automaton.set_initial_state("u0")
        automaton.set_accept_state("u_acc")
        return automaton


class OfficeWorldDeliverCoffeeOrMailEnv(OfficeWorldEnv):
    """
    Deliver coffee or the mail to the office while avoiding the plants.
    """
    def is_goal_achieved(self):
        return (self.has_coffee or self.has_mail) and self.is_agent_at_office()

    def get_restricted_observables(self):
        return [OfficeWorldObject.COFFEE, OfficeWorldObject.MAIL, OfficeWorldObject.OFFICE, OfficeWorldObject.PLANT]

    def get_automaton(self):
        automaton = SubgoalAutomaton()
        automaton.add_state("u0")
        automaton.add_state("u1")
        automaton.add_state("u_acc")

        automaton.add_edge("u0", "u_acc", [OfficeWorldObject.COFFEE, OfficeWorldObject.OFFICE])
        automaton.add_edge("u0", "u_acc", [OfficeWorldObject.MAIL, OfficeWorldObject.OFFICE])
        automaton.add_edge("u1", "u_acc", [OfficeWorldObject.OFFICE, "~" + OfficeWorldObject.PLANT])

        if self.drop_coffee_on_plant_enable:
            automaton.add_state("u2")
            automaton.add_edge("u0", "u1", [OfficeWorldObject.COFFEE, "~" + OfficeWorldObject.OFFICE])
            automaton.add_edge("u0", "u2", [OfficeWorldObject.MAIL, "~" + OfficeWorldObject.OFFICE])
            automaton.add_edge("u1", "u0", [OfficeWorldObject.PLANT])
            automaton.add_edge("u2", "u_acc", [OfficeWorldObject.OFFICE])
        else:
            automaton.add_state("u_rej")
            automaton.set_reject_state("u_rej")
            automaton.add_edge("u0", "u1", [OfficeWorldObject.COFFEE, "~" + OfficeWorldObject.OFFICE])
            automaton.add_edge("u0", "u1", [OfficeWorldObject.MAIL, "~" + OfficeWorldObject.OFFICE])
            automaton.add_edge("u0", "u_rej", [OfficeWorldObject.PLANT, "~" + OfficeWorldObject.COFFEE, "~" + OfficeWorldObject.MAIL])
            automaton.add_edge("u1", "u_rej", [OfficeWorldObject.PLANT])

        automaton.set_initial_state("u0")
        automaton.set_accept_state("u_acc")
        return automaton


class OfficeWorldPatrolABEnv(OfficeWorldEnv):
    """
    Visit rooms A and B (in that order) while avoiding the plants.
    """
    def is_goal_achieved(self):
        if self.agent in self.locations:
            return self.locations[self.agent] == OfficeWorldObject.ROOM_B and \
                   self.visited_rooms == OfficeWorldRoomVisits.VISITED_A
        return False

    def get_restricted_observables(self):
        return [OfficeWorldObject.PLANT, OfficeWorldObject.ROOM_A, OfficeWorldObject.ROOM_B]

    def get_automaton(self):
        automaton = SubgoalAutomaton()
        automaton.add_state("u0")
        automaton.add_state("u1")
        automaton.add_state("u_acc")
        automaton.add_state("u_rej")
        automaton.add_edge("u0", "u1", [OfficeWorldObject.ROOM_A, "~" + OfficeWorldObject.PLANT])
        automaton.add_edge("u0", "u_rej", [OfficeWorldObject.PLANT])
        automaton.add_edge("u1", "u_acc", [OfficeWorldObject.ROOM_B, "~" + OfficeWorldObject.PLANT])
        automaton.add_edge("u1", "u_rej", [OfficeWorldObject.PLANT])
        automaton.set_initial_state("u0")
        automaton.set_accept_state("u_acc")
        automaton.set_reject_state("u_rej")
        return automaton


class OfficeWorldPatrolABStrictEnv(OfficeWorldPatrolABEnv):
    """
    Visit rooms A and B (in that order) while avoiding the plants. The order is *strict*, i.e. B cannot be visited
    before A.
    """
    def is_deadend(self):
        if self.agent in self.locations:
            location = self.locations[self.agent]
            if (self.visited_rooms == OfficeWorldRoomVisits.VISITED_NONE and location in [OfficeWorldObject.ROOM_B]) or \
               (self.visited_rooms == OfficeWorldRoomVisits.VISITED_A and location in [OfficeWorldObject.ROOM_A]):
                return True
        return super().is_deadend()

    def get_automaton(self):
        automaton = super().get_automaton()
        automaton.add_edge("u0", "u_rej", [OfficeWorldObject.ROOM_B, "~" + OfficeWorldObject.ROOM_A])
        automaton.add_edge("u1", "u_rej", [OfficeWorldObject.ROOM_A, "~" + OfficeWorldObject.ROOM_B])
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
        automaton.add_state("u0")
        automaton.add_state("u1")
        automaton.add_state("u2")
        automaton.add_state("u_acc")
        automaton.add_state("u_rej")
        automaton.add_edge("u0", "u1", [OfficeWorldObject.ROOM_A, "~" + OfficeWorldObject.PLANT])
        automaton.add_edge("u0", "u_rej", [OfficeWorldObject.PLANT])
        automaton.add_edge("u1", "u2", [OfficeWorldObject.ROOM_B, "~" + OfficeWorldObject.PLANT])
        automaton.add_edge("u1", "u_rej", [OfficeWorldObject.PLANT])
        automaton.add_edge("u2", "u_acc", [OfficeWorldObject.ROOM_C, "~" + OfficeWorldObject.PLANT])
        automaton.add_edge("u2", "u_rej", [OfficeWorldObject.PLANT])
        automaton.set_initial_state("u0")
        automaton.set_accept_state("u_acc")
        automaton.set_reject_state("u_rej")
        return automaton


class OfficeWorldPatrolABCStrictEnv(OfficeWorldPatrolABCEnv):
    """
    Visit rooms A, B and C (in that order) while avoiding the plants. The order is *strict*. For example, B cannot be
    visited before A, and A cannot be visited after visiting B.
    """
    def is_deadend(self):
        if self.agent in self.locations:
            location = self.locations[self.agent]
            if (self.visited_rooms == OfficeWorldRoomVisits.VISITED_NONE and location in [OfficeWorldObject.ROOM_B,
                                                                                          OfficeWorldObject.ROOM_C]) or \
               (self.visited_rooms == OfficeWorldRoomVisits.VISITED_A and location in [OfficeWorldObject.ROOM_A,
                                                                                       OfficeWorldObject.ROOM_C]) or \
               (self.visited_rooms == OfficeWorldRoomVisits.VISITED_AB and location in [OfficeWorldObject.ROOM_A,
                                                                                        OfficeWorldObject.ROOM_B]):
                return True
        return super().is_deadend()

    def get_automaton(self):
        automaton = super().get_automaton()
        automaton.add_edge("u0", "u_rej", [OfficeWorldObject.ROOM_B, "~" + OfficeWorldObject.ROOM_A])
        automaton.add_edge("u0", "u_rej", [OfficeWorldObject.ROOM_C, "~" + OfficeWorldObject.ROOM_A])
        automaton.add_edge("u1", "u_rej", [OfficeWorldObject.ROOM_A, "~" + OfficeWorldObject.ROOM_B])
        automaton.add_edge("u1", "u_rej", [OfficeWorldObject.ROOM_C, "~" + OfficeWorldObject.ROOM_B])
        automaton.add_edge("u2", "u_rej", [OfficeWorldObject.ROOM_A, "~" + OfficeWorldObject.ROOM_C])
        automaton.add_edge("u2", "u_rej", [OfficeWorldObject.ROOM_B, "~" + OfficeWorldObject.ROOM_C])
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
        automaton.add_state("u0")
        automaton.add_state("u1")
        automaton.add_state("u2")
        automaton.add_state("u3")
        automaton.add_state("u_acc")
        automaton.add_state("u_rej")
        automaton.add_edge("u0", "u1", [OfficeWorldObject.ROOM_A, "~" + OfficeWorldObject.PLANT])
        automaton.add_edge("u0", "u_rej", [OfficeWorldObject.PLANT])
        automaton.add_edge("u1", "u2", [OfficeWorldObject.ROOM_B, "~" + OfficeWorldObject.PLANT])
        automaton.add_edge("u1", "u_rej", [OfficeWorldObject.PLANT])
        automaton.add_edge("u2", "u3", [OfficeWorldObject.ROOM_C, "~" + OfficeWorldObject.PLANT])
        automaton.add_edge("u2", "u_rej", [OfficeWorldObject.PLANT])
        automaton.add_edge("u3", "u_acc", [OfficeWorldObject.ROOM_D, "~" + OfficeWorldObject.PLANT])
        automaton.add_edge("u3", "u_rej", [OfficeWorldObject.PLANT])
        automaton.set_initial_state("u0")
        automaton.set_accept_state("u_acc")
        automaton.set_reject_state("u_rej")
        return automaton


class OfficeWorldPatrolABCDStrictEnv(OfficeWorldPatrolABCDEnv):
    """
    Visit rooms A, B, C and D (in that order) while avoiding the plants. The order is *strict*. For example, B cannot be
    visited before A, and A cannot be visited after visiting B.
    """
    def is_deadend(self):
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
        return super().is_deadend()

    def get_automaton(self):
        automaton = super().get_automaton()
        automaton.add_edge("u0", "u_rej", [OfficeWorldObject.ROOM_B, "~" + OfficeWorldObject.ROOM_A])
        automaton.add_edge("u0", "u_rej", [OfficeWorldObject.ROOM_C, "~" + OfficeWorldObject.ROOM_A])
        automaton.add_edge("u0", "u_rej", [OfficeWorldObject.ROOM_D, "~" + OfficeWorldObject.ROOM_A])
        automaton.add_edge("u1", "u_rej", [OfficeWorldObject.ROOM_A, "~" + OfficeWorldObject.ROOM_B])
        automaton.add_edge("u1", "u_rej", [OfficeWorldObject.ROOM_C, "~" + OfficeWorldObject.ROOM_B])
        automaton.add_edge("u1", "u_rej", [OfficeWorldObject.ROOM_D, "~" + OfficeWorldObject.ROOM_B])
        automaton.add_edge("u2", "u_rej", [OfficeWorldObject.ROOM_A, "~" + OfficeWorldObject.ROOM_C])
        automaton.add_edge("u2", "u_rej", [OfficeWorldObject.ROOM_B, "~" + OfficeWorldObject.ROOM_C])
        automaton.add_edge("u2", "u_rej", [OfficeWorldObject.ROOM_D, "~" + OfficeWorldObject.ROOM_C])
        automaton.add_edge("u3", "u_rej", [OfficeWorldObject.ROOM_A, "~" + OfficeWorldObject.ROOM_D])
        automaton.add_edge("u3", "u_rej", [OfficeWorldObject.ROOM_B, "~" + OfficeWorldObject.ROOM_D])
        automaton.add_edge("u3", "u_rej", [OfficeWorldObject.ROOM_C, "~" + OfficeWorldObject.ROOM_D])
        return automaton
