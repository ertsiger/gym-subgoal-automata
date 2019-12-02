import os
import subprocess


class SubgoalAutomaton:
    """
    Class used for modelling the subgoal automata as deterministic finite automata (DFA). It provides methods for
    building the graph incrementally by adding states and labelled edges. It also allows to export a given automaton
    as an image using Graphviz.
    """
    GRAPHVIZ_SUBPROCESS_TXT = "diagram.txt"  # name of the file temporally used to store the DFA in Graphviz format

    def __init__(self):
        self.states = set()
        self.edges = {}
        self.initial_state = None
        self.accept_state = None
        self.reject_state = None
        self.min_distance_matrix = None

    def add_state(self, state):
        """Adds a state to the set of states and creates an entry in the set of edges that go from that state."""
        if state not in self.states:
            self.states.add(state)
            self.edges[state] = []

        self.min_distance_matrix = None  # reset min distance matrix (we don't want to return incorrect results!)

    def get_states(self):
        """Returns the set of states."""
        return self.states

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
        self.edges[from_state].append((conditions, to_state))

        self.min_distance_matrix = None  # reset min distance matrix (we don't want to return incorrect results!)

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
                collapsed_and_edges[from_state][to_state].append("&".join(cond))

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

    def get_next_states(self, current_state, observations):
        """
        Returns a set of possible next states given the current state and the current observations. If no condition
        to a next state holds, then the next state will be the current state.

        Note that we learn deterministic finite automata with reference to a set of examples. Thus, the resulting
        automaton will be deterministic for that set of examples, but might not be for next examples. This is the
        reason why we return a set of states instead of a single state. The class using the DFA class will have to
        check for the number of next states in order to enforce a deterministic automaton in the following learning
        steps.
        """
        next_states = set([])

        for condition, cand_state in self.edges[current_state]:
            if self._is_condition_satisfied(condition, observations):
                next_states.add(cand_state)

        if len(next_states) == 0:  # if no condition held, the next state is the current state
            next_states.add(current_state)

        return next_states

    def _is_condition_satisfied(self, condition, observations):
        """Returns true if a condition is satisfied by a set of observations. If a condition is empty, it is always
        true regardless of the observations."""
        if len(condition) == 0:  # empty conditions = unconditional transition (always taken)
            return True

        # check if some condition in the array does not hold (conditions are AND)
        for literal in condition:
            if literal.startswith("~"):
                fluent = literal[1:]  # take literal without the tilde
                if fluent in observations:
                    return False
            else:
                fluent = literal
                if fluent not in observations:
                    return False
        return True

    def get_distance(self, from_state, to_state):
        if self.min_distance_matrix is None:
            self._compute_min_distance_matrix()
        return self.min_distance_matrix[from_state][to_state]

    def get_distance_to_accept_state(self, from_state):
        if self.has_accept_state():
            return self.get_distance(from_state, self.accept_state)
        else:
            raise RuntimeError("Error: The automaton does not have an accepting state!")

    def _compute_min_distance_matrix(self):
        """Computes the minimum distance from every node to every node in the DFA. When a node cannot be reached, the
        distance is float('inf')."""
        self.min_distance_matrix = {}
        for state in self.get_states():
            self.min_distance_matrix[state] = self._compute_min_distances_from_state(state)

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

# Usage example
# if __name__ == "__main__":
#     dfa = SubgoalAutomaton()
#     dfa.add_state("u0")
#     dfa.add_state("u1")
#     dfa.add_state("u2")
#     dfa.add_edge("u0", "u1", ["a", "b"])
#     dfa.add_edge("u0", "u1", ["c"])
#     dfa.add_edge("u2", "u1", ["~d"])
#     dfa.add_edge("u1", "u2", ["e", "f"])
#     dfa.plot(".", "output.png")
#
#     print(dfa.get_next_states("u0", ["c"]))
#     print(dfa.get_next_states("u0", ["b", "a"]))
#     print(dfa.get_next_states("u2", ["d"]))
#     print(dfa.get_distance("u1", "u0"))

