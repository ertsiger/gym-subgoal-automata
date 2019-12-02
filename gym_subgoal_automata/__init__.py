from gym.envs.registration import register

# OfficeWorld
register(
    id='OfficeWorldDeliverCoffee-v0',
    entry_point='gym_subgoal_automata.envs.officeworld:OfficeWorldDeliverCoffeeEnv',
)

register(
    id='OfficeWorldDeliverCoffeeAndMail-v0',
    entry_point='gym_subgoal_automata.envs.officeworld:OfficeWorldDeliverCoffeeAndMailEnv',
)

register(
    id='OfficeWorldPatrolABC-v0',
    entry_point='gym_subgoal_automata.envs.officeworld:OfficeWorldPatrolABCEnv',
)

register(
    id='OfficeWorldPatrolABCStrict-v0',
    entry_point='gym_subgoal_automata.envs.officeworld:OfficeWorldPatrolABCStrictEnv',
)

register(
    id='OfficeWorldPatrolABCD-v0',
    entry_point='gym_subgoal_automata.envs.officeworld:OfficeWorldPatrolABCDEnv',
)

register(
    id='OfficeWorldPatrolABCDStrict-v0',
    entry_point='gym_subgoal_automata.envs.officeworld:OfficeWorldPatrolABCDStrictEnv',
)

# WaterWorld
register(
    id='WaterWorld-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldEnv',
)

register(
    id='WaterWorldRedGreen-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldRedGreenEnv'
)

register(
    id='WaterWorldBlueCyan-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldBlueCyanEnv'
)

register(
    id='WaterWorldMagentaYellow-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldMagentaYellowEnv'
)

register(
    id='WaterWorldRedGreenAndBlueCyan-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldRedGreenAndBlueCyanEnv'
)

register(
    id='WaterWorldBlueCyanAndMagentaYellow-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldBlueCyanAndMagentaYellowEnv'
)

register(
    id='WaterWorldRedGreenAndMagentaYellow-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldRedGreenAndMagentaYellowEnv'
)

register(
    id='WaterWorldRedGreenAndBlueCyanAndMagentaYellow-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldRedGreenAndBlueCyanAndMagentaYellowEnv'
)

register(
    id='WaterWorldRedGreenBlueAndCyanMagentaYellow-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldRedGreenBlueAndCyanMagentaYellowEnv'
)

register(
    id='WaterWorldRedGreenBlueStrict-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldRedGreenBlueStrictEnv'
)

register(
    id='WaterWorldCyanMagentaYellowStrict-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldCyanMagentaYellowStrictEnv'
)

register(
    id='WaterWorldRedAndBlueCyan-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldRedAndBlueCyanEnv'
)

register(
    id='WaterWorldRedGreenAndBlue-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldRedGreenAndBlueEnv'
)

register(
    id='WaterWorldRedGreenBlue-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldRedGreenBlueEnv'
)

register(
    id='WaterWorldRedGreenBlueCyan-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldRedGreenBlueCyanEnv'
)
