import os
import subprocess
from gym_subgoal_automata.utils.condition import EdgeCondition


class MultipleConditionsHoldException(Exception):
    def __init__(self):
        super().__init__("Error: Multiple conditions cannot hold at the same time.")


class SubgoalAutomaton:
    """
    Class used for modelling the subgoal automata as deterministic finite automata (DFA). It provides methods for
    building the graph incrementally by adding states and labelled edges. It also allows to export a given automaton
    as an image using Graphviz.
    """
    GRAPHVIZ_SUBPROCESS_TXT = "diagram.txt"  # name of the file temporally used to store the DFA in Graphviz format

    def __init__(self):
        self.states = []
        self.edges = {}
        self.initial_state = None
        self.accept_state = None
        self.reject_state = None
        self.distance_matrix = None

    def add_state(self, state):
        """Adds a state to the set of states and creates an entry in the set of edges that go from that state."""
        if state not in self.states:
            self.states.append(state)
            self.states.sort()
            self.edges[state] = []

        self.distance_matrix = None  # reset min distance matrix (we don't want to return incorrect results!)

    def get_states(self):
        """Returns the set of states."""
        return self.states

    def get_state_id(self, state):
        return self.states.index(state)

    def get_num_states(self):
        """Returns the number of states in the automaton."""
        return len(self.states)

    def add_edge(self, from_state, to_state, conditions):
        """
        Adds an edge to the set of edges. Note that the set of edges is (from_state) -> list((condition, to_state)).
        If the states are not in the set of states, they are added.
        """
        if from_state not in self.edges:
            self.add_state(from_state)
        if to_state not in self.edges:
            self.add_state(to_state)

        condition = EdgeCondition(tuple(sorted(conditions)))  # keep all conditions sorted
        self.edges[from_state].append((condition, to_state))

        self.distance_matrix = None  # reset min distance matrix (we don't want to return incorrect results!)

    def set_initial_state(self, state):
        """Sets the initial state (there can only be one initial state)."""
        self.initial_state = state

    def get_initial_state(self):
        """Returns the name of the initial state."""
        return self.initial_state

    def set_accept_state(self, state):
        """Sets a given state as the accepting state."""
        self.accept_state = state

    def is_accept_state(self, state):
        """Tells whether a given state is the accepting state."""
        return state == self.accept_state

    def has_accept_state(self):
        """Returns true if the automaton contains an accepting state."""
        return self.accept_state is not None

    def set_reject_state(self, state):
        """Sets a given state as the rejecting states."""
        self.reject_state = state

    def is_reject_state(self, state):
        """Tells whether a given state is the rejecting state."""
        return state == self.reject_state

    def has_reject_state(self):
        """Returns true if the automaton contains a rejecting state."""
        return self.reject_state is not None

    def is_terminal_state(self, state):
        """Tells whether a given state is terminal (accepting or rejecting state)."""
        return self.is_accept_state(state) or self.is_reject_state(state)

    def is_accepting_transition(self, from_state, to_state):
        """Returns True the 'from' state is not accepting and the 'to' state is an accepting state."""
        return not self.is_accept_state(from_state) and self.is_accept_state(to_state)

    def is_rejecting_transition(self, from_state, to_state):
        """Returns True the 'from' state is not rejecting and the 'to' state is a rejecting state."""
        return not self.is_reject_state(from_state) and self.is_reject_state(to_state)

    def is_terminal_transition(self, from_state, to_state):
        """Returns True the 'from' state is not terminal and the 'to' state is a terminal state."""
        return self.is_accepting_transition(from_state, to_state) or self.is_rejecting_transition(from_state, to_state)

    def plot(self, plot_folder, filename, use_subprocess=True):
        """
        Plots the DFA into a file.

        Keyword arguments:
            plot_folder -- folder where the plot will be written.
            filename -- name of the file containing the plot.
            use_subprocess -- if True, it runs Graphviz from the command line; else, it runs graphviz from the Python
                              package.
        """
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        solution_plot_path = os.path.join(plot_folder, filename)

        if use_subprocess:
            diagram_path = os.path.join(plot_folder, SubgoalAutomaton.GRAPHVIZ_SUBPROCESS_TXT)
            self._write_graphviz_diagram(diagram_path)
            self._run_graphviz_subprocess(diagram_path, solution_plot_path)
        else:
            self._run_graphviz_api(solution_plot_path)

    def _write_graphviz_diagram(self, diagram_path):
        """Exports DFA into a file using Graphviz format."""
        graphviz_str = "digraph G {\n"

        # write states
        for state in self.states:
            graphviz_str += "%s [label=\"%s\"];\n" % (state, state)

        # write edges - collapsed edges means compressing OR conditions into single edges labelled with an OR in the
        #               middle
        collapsed_edges = self._get_collapsed_edges()
        for from_state, to_state, condition in collapsed_edges:
            graphviz_str += "%s -> %s [label=\"%s\"];\n" % (from_state, to_state, condition)

        graphviz_str += "}"

        with open(diagram_path, 'w') as f:
            f.write(graphviz_str)

    def _get_collapsed_edges(self):
        """Returns the current set of edges but collapsed. This means joining different edges between two states into
        a single edge to have a cleaner output."""

        # collapse AND conditions first (have them in a readable format)
        collapsed_and_edges = {}
        for from_state in self.edges:
            collapsed_and_edges[from_state] = {}
            for cond, to_state in self.edges[from_state]:
                if to_state not in collapsed_and_edges[from_state]:
                    collapsed_and_edges[from_state][to_state] = []
                collapsed_and_edges[from_state][to_state].append(str(cond))

        # collapse OR conditions
        collapsed_edges = []
        for from_state in collapsed_and_edges:
            for to_state in collapsed_and_edges[from_state]:
                condition = ""
                conditions = collapsed_and_edges[from_state][to_state]
                if len(conditions) == 1:  # no OR involved
                    condition = collapsed_and_edges[from_state][to_state][0]
                else:
                    for i in range(0, len(conditions)):
                        if i > 0:
                            condition += "|"
                        condition += "(" + conditions[i] + ")"
                collapsed_edges.append((from_state, to_state, condition))

        return collapsed_edges

    def _run_graphviz_subprocess(self, diagram_path, filename):
        """
        Runs Graphviz from the command line and exports the automaton in .png format.

        Keyword arguments:
            diagram_path -- path to the .txt file containing the automaton in Graphviz format.
            filename -- output file name.
        """
        subprocess.call(["dot",
                         "-Tpng",
                         diagram_path,
                         "-o",
                         filename])

    def _run_graphviz_api(self, filename):
        """Runs Graphviz using the Python package and exports the figure in the file specified by filename."""
        from graphviz import Digraph

        dot = Digraph()

        # create Graphviz states
        for state in self.states:
            dot.node(state)

        # create Graphviz edges
        collapsed_edges = self._get_collapsed_edges()
        for from_state, to_state, condition in collapsed_edges:
            dot.edge(from_state, to_state, condition)

        # render the resulting Graphviz graph
        dot.render(filename)

    def get_next_state(self, current_state, observations):
        """
        Returns the next state given the current state and the current observations. If no condition
        to a next state holds, then the next state will be the current state.
        """
        candidate_states = set()
        for condition, candidate_state in self.edges[current_state]:
            if condition.is_satisfied(observations):
                candidate_states.add(candidate_state)

        if len(candidate_states) > 1:
            raise MultipleConditionsHoldException
        elif len(candidate_states) == 1:
            return candidate_states.pop()
        return current_state

    def get_distance(self, from_state, to_state, method):
        if self.distance_matrix is None:
            self._compute_distance_matrix(method)
        return self.distance_matrix[from_state][to_state]

    def get_distance_to_accept_state(self, from_state, method="min_distance"):
        if self.has_accept_state():
            return self.get_distance(from_state, self.accept_state, method)
        else:
            raise RuntimeError("Error: The automaton does not have an accepting state!")

    def _compute_distance_matrix(self, method):
        """Computes the minimum distance from every node to every node in the DFA. When a node cannot be reached, the
        distance is float('inf')."""
        self.distance_matrix = {}

        if method == "min_distance":
            method_function = self._compute_min_distances_from_state
        elif method == "max_distance":
            method_function = self._compute_max_distances_from_state
        else:
            raise RuntimeError("Error: Unknown method '%s' for computing the distance matrix in the automaton." % method)

        for state in self.get_states():
            self.distance_matrix[state] = method_function(state)

    def _compute_min_distances_from_state(self, state):
        """Computes the minimum distance from one state to all the other states."""
        distances = {s: float("inf") for s in self.get_states()}
        distances[state] = 0

        s_queue = [(state, 0)]
        visited = {state}

        while len(s_queue) > 0:
            current_state, current_distance = s_queue.pop()

            for _, next_state in self.edges[current_state]:
                if next_state not in visited:
                    s_queue.append((next_state, current_distance + 1))
                    visited.add(next_state)
                    distances[next_state] = current_distance + 1

        return distances

    def _compute_max_distances_from_state(self, state):
        """Computes the maximum distance from one state to all the other states. IMPORTANT: this is limited to acyclic
        graphs!"""
        distances = {s: float("-inf") for s in self.get_states()}
        distances[state] = 0
        top_order = self._get_topological_order()
        for state in top_order:
            for _, next_state in self.edges[state]:
                if distances[next_state] < distances[state] + 1:
                    distances[next_state] = distances[state] + 1
        for s in distances:
            if distances[s] == float('-inf'):
                distances[s] = float('inf')
        return distances

    def _get_topological_order(self):
        stack = [self.initial_state]
        indegrees = self._get_in_degrees()
        top_order = []

        while len(stack) > 0:
            state = stack.pop()
            top_order.append(state)
            for _, next_state in self.edges[state]:
                indegrees[next_state] -= 1
                if indegrees[next_state] == 0:
                    stack.append(next_state)

        return top_order

    def _get_in_degrees(self):
        degrees = {s : 0 for s in self.states}
        for state in self.states:
            checked_states = set()
            for _, next_state in self.edges[state]:
                if next_state not in checked_states:
                    degrees[next_state] += 1
                    checked_states.add(next_state)
        return degrees

    def get_all_conditions(self):
        """Returns all conditions contained in the automaton without repetitions."""
        all_conditions = set()
        for state in self.get_states():
            for condition, _ in self.edges[state]:
                all_conditions.add(condition)
        return sorted(list(all_conditions))

    def get_num_unique_conditions(self):
        return len(self.get_all_conditions())

    def get_num_outgoing_edges(self, state):
        """Returns the number of outgoing edges that a given state has."""
        return len(self.edges[state])

    def get_outgoing_conditions(self, state):
        return [self.get_condition(state, edge_id) for edge_id in range(self.get_num_outgoing_edges(state))]

    def get_outgoing_condition_id(self, state, condition):
        return self.get_outgoing_conditions(state).index(condition)

    def get_condition(self, state, edge_id):
        """Returns the condition for an edge of a given state."""
        return self.edges[state][edge_id][0]


# Usage example
if __name__ == "__main__":
    dfa = SubgoalAutomaton()
    dfa.add_state("u0")
    dfa.add_state("u1")
    dfa.add_state("uA")
    dfa.add_state("uR")
    dfa.set_initial_state("u0")
    dfa.set_accept_state("uA")
    dfa.set_reject_state("uR")

    dfa.add_edge("u0", "u1", ["f", "~g"])
    dfa.add_edge("u0", "uA", ["f", "g"])
    dfa.add_edge("u0", "uR", ["n", "~f", "~g"])
    dfa.add_edge("u1", "uA", ["g"])
    dfa.add_edge("u1", "uR", ["n", "~g"])

    # dfa.plot(".", "output.png")

    print("(u0, [f]) -->", dfa.get_next_state("u0", ["f"]))
    print("(u0, [a]) -->", dfa.get_next_state("u0", ["a"]))
    print("(u0, [g,f]) -->", dfa.get_next_state("u0", ["g", "f"]))
    print("(u0, [n]) -->", dfa.get_next_state("u0", ["n"]))
    print("(u1, [g,a,b]) -->", dfa.get_next_state("u1", ["g", "a", "b"]))
    print("(u1, [n]) -->", dfa.get_next_state("u1", ["n"]))

    print("min_dist(u0, uA) =", dfa.get_distance("u0", "uA", "min_distance"))
    # print("max_dist(u0, uA) =", dfa.get_distance("u0", "uA", "max_distance"))

    print("Conditions:", dfa.get_all_conditions())
