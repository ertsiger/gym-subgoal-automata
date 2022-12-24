# Gym Subgoal Automata
Environments from [[Toro Icarte et al., 2018]](#references) using OpenAI Gym API. This repository currently complements
the code for [[Furelos-Blanco et al., 2020]](#references) and [[Furelos-Blanco et al., 2021]](#references), 
whose code is [here](https://github.com/ertsiger/induction-subgoal-automata-rl).

1. [Installation](#installation)
1. [Usage](#usage)
1. [Acknowledgments](#acks)
1. [References](#references)

## <a name="installation"></a>Installation
To install the package, you just have to clone the repository and run the following commands:
```
cd gym-subgoal-automata
pip install -e .
```

We recommend you to use a [virtual environment](https://docs.python.org/3/tutorial/venv.html) since the requirements of 
this package may affect your current installation. The `setup.py` file contains the current requirements for the code to run safely.

The learned subgoal automata are exported to `.png` using Graphviz. You can follow the instructions in the [official webpage](https://graphviz.org/download/) to install it.

## <a name="usage"></a>Usage
The repository has implementations for the OfficeWorld and WaterWorld environments and different associated success/fail tasks. 
You can find the list of all tasks in the file `gym_subgoal_automata/__init__.py`. The following is an example of how to
instantiate the OfficeWorld environment where the task is "deliver coffee to the office".

```
import gym
env = gym.make("gym_subgoal_automata:OfficeWorldDeliverCoffee-v0", params={"generation": "random", "environment_seed": 0})
```

You can use the method `env.play()` to use the environment with your keyboard using the `w`, `a`, `s` and `d` keys. In this task you have to observe `f` (coffee) and then `g` (office) while avoiding the `n` (plants/decorations).

## <a name="acks"></a>Acknowledgments
We thank the authors of reward machines for open sourcing their [code](https://bitbucket.org/RToroIcarte/qrm). The code 
in this repository is heavily based on theirs.

## <a name="references"></a>References
* Toro Icarte, R.; Klassen, T. Q.; Valenzano, R. A.; and McIlraith, S. A. 2018. [_Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning_](http://proceedings.mlr.press/v80/icarte18a.html). Proceedings of the 35th International Conference on Machine Learning.
* Furelos-Blanco, D.; Law, M.; Russo, A.; Broda, K.; and Jonsson, A. 2020. [_Induction of Subgoal Automata for Reinforcement Learning_](https://doi.org/10.1609/aaai.v34i04.5802). Proceedings of the 34th AAAI Conference on Artificial Intelligence.
* Furelos-Blanco, D.; Law, M.; Jonsson, A.; Broda, K.; and Russo, A. 2021. [_Induction and Exploitation of Subgoal Automata for Reinforcement Learning_](https://jair.org/index.php/jair/article/view/12372). J. Artif. Intell. Res., 70, 1031-1116.

