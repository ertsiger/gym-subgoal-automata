from collections import namedtuple

Condition = namedtuple("Condition", "condition")


class EdgeCondition(Condition):
    """Returns true if a condition is satisfied by a set of observations. If a condition is empty, it is always
    true regardless of the observations."""
    def is_satisfied(self, observations):
        if len(self.condition) == 0:  # empty conditions = unconditional transition (always taken)
            return True

        # check if some condition in the array does not hold (conditions are AND)
        for literal in self.condition:
            if literal.startswith("~"):
                fluent = literal[1:]  # take literal without the tilde
                if fluent in observations:
                    return False
            else:
                fluent = literal
                if fluent not in observations:
                    return False
        return True

    def get_as_sorted_tuple(self):
        return tuple(sorted(self.condition))

    def __str__(self):
        return "&".join(self.condition)

    def is_subsumed(self, condition_p):
        condition_set, condition_p_set = set(self.condition), set(condition_p)
        return condition_set.issubset(condition_p_set)

    def subsumes(self, condition_p):
        condition_set, condition_p_set = set(self.condition), set(condition_p)
        return condition_set.issuperset(condition_p_set)

    def get_num_matching_symbols(self, condition_p):
        condition_set, condition_p_set = set(self.condition), set(condition_p.condition)
        return len(condition_set.intersection(condition_p_set))

    def get_num_positive_matching_symbols(self, condition_p):
        pos_conditions , pos_conditions_p = set(self.get_positive_conditions()), set(condition_p.get_positive_conditions())
        return len(pos_conditions.intersection(pos_conditions_p))

    def get_positive_conditions(self):
        return [x for x in self.condition if not x.startswith("~")]
