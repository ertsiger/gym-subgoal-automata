from gym.envs.registration import register

# OfficeWorld
register(
    id='OfficeWorldDeliverCoffee-v0',
    entry_point='gym_subgoal_automata.envs.officeworld:OfficeWorldDeliverCoffeeEnv',
)

register(
    id='OfficeWorldDeliverMail-v0',
    entry_point='gym_subgoal_automata.envs.officeworld:OfficeWorldDeliverMailEnv',
)

register(
    id='OfficeWorldDeliverCoffeeAndMail-v0',
    entry_point='gym_subgoal_automata.envs.officeworld:OfficeWorldDeliverCoffeeAndMailEnv',
)

register(
    id='OfficeWorldDeliverCoffeeOrMail-v0',
    entry_point='gym_subgoal_automata.envs.officeworld:OfficeWorldDeliverCoffeeOrMailEnv',
)

register(
    id='OfficeWorldPatrolAB-v0',
    entry_point='gym_subgoal_automata.envs.officeworld:OfficeWorldPatrolABEnv',
)

register(
    id='OfficeWorldPatrolABStrict-v0',
    entry_point='gym_subgoal_automata.envs.officeworld:OfficeWorldPatrolABStrictEnv',
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
    id='WaterWorldRedGreenStrict-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldRedGreenStrictEnv'
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

register(
    id='WaterWorldRedGreenJoint-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldRedGreenJointEnv'
)

register(
    id='WaterWorldRedAndGreenAndBlue-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldRedAndGreenAndBlueEnv'
)

register(
    id='WaterWorldRedGreenBlueCyanYellow-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldRedGreenBlueCyanYellowEnv'
)

register(
    id='WaterWorldRedGreenAvoidMagenta-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldRedGreenAvoidMagentaEnv'
)

register(
    id='WaterWorldRedGreenAvoidMagentaYellow-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldRedGreenAvoidMagentaYellowEnv'
)

register(
    id='WaterWorldRedAvoidMagenta-v0',
    entry_point='gym_subgoal_automata.envs.waterworld:WaterWorldRedAvoidMagentaEnv'
)

# CraftWorld
register(
    id='CraftWorldMakePlank-v0',
    entry_point='gym_subgoal_automata.envs.craftworld:CraftWorldMakePlankEnv',
)

register(
    id='CraftWorldMakeStick-v0',
    entry_point='gym_subgoal_automata.envs.craftworld:CraftWorldMakeStickEnv',
)

register(
    id='CraftWorldMakeCloth-v0',
    entry_point='gym_subgoal_automata.envs.craftworld:CraftWorldMakeClothEnv',
)

register(
    id='CraftWorldMakeRope-v0',
    entry_point='gym_subgoal_automata.envs.craftworld:CraftWorldMakeRopeEnv',
)

register(
    id='CraftWorldMakeBridge-v0',
    entry_point='gym_subgoal_automata.envs.craftworld:CraftWorldMakeBridgeEnv',
)

register(
    id='CraftWorldMakeBed-v0',
    entry_point='gym_subgoal_automata.envs.craftworld:CraftWorldMakeBedEnv',
)

register(
    id='CraftWorldMakeAxe-v0',
    entry_point='gym_subgoal_automata.envs.craftworld:CraftWorldMakeAxeEnv',
)

register(
    id='CraftWorldMakeShears-v0',
    entry_point='gym_subgoal_automata.envs.craftworld:CraftWorldMakeShearsEnv',
)

register(
    id='CraftWorldGetGold-v0',
    entry_point='gym_subgoal_automata.envs.craftworld:CraftWorldGetGoldEnv',
)

register(
    id='CraftWorldGetGem-v0',
    entry_point='gym_subgoal_automata.envs.craftworld:CraftWorldGetGemEnv',
)
