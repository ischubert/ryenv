# %%
"""
Tests for the disk env class
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ryenv

# %%
MYENV = ryenv.DiskEnv()
MYENV.view()

# %%
CHANGE = MYENV.transition((-0.3, -0.2), fps=30)
print(CHANGE)

# %%
# Test weird anisotropic behaviour

for center_position in 1.2*np.random.rand(
    10, 2
) - 0.6:
    angles = np.linspace(0, 2*np.pi, num=32, endpoint=False)
    states = center_position[None, :] + np.array([
        [0.31 * np.cos(angle), 0.31 * np.sin(angle)]
        for angle in angles
    ])
    actions = np.array([
        [-MYENV.action_length *
            np.cos(angle), -MYENV.action_length * np.sin(angle)]
        for angle in angles
    ])
    goals = np.array([
        [-np.cos(angle), -np.sin(angle)]
        for angle in angles
    ])

    changes = []
    rewards = []
    for state, action, goal in zip(states, actions, goals):
        MYENV.reset(state, disk_position=center_position)
        change = MYENV.transition(action) - center_position
        changes.append(change.copy())
        reward = MYENV.calculate_reward(change, goal)
        rewards.append(reward)

    rewards = np.array(rewards)

    plt.figure(figsize=(3, 3))
    ax = plt.gca()
    ax.add_artist(mpatches.Circle((0, 0), 0.25, color='#C7C7D3'))
    states_ind = 1
    ax.add_artist(mpatches.Circle(
        (states[states_ind, 0], states[states_ind, 1]), 0.06, color='#8F8F8F'))
    for ind, [state, action, change] in enumerate(zip(
        states - center_position[None, :],
        actions,
        changes
    )):
        # if not ind == states_ind:
        plt.plot(state[0], state[1], 'o', color='#8F8F8F')
        plt.arrow(
            state[0], state[1], change[0], change[1],
            head_width=0.01, color='#2660E8'
        )
        plt.arrow(
            state[0], state[1], action[0], action[1],
            head_width=0.01, color=np.array((223, 41, 53, 255))/255
        )

    plt.xlim(-0.37, 0.37)
    plt.ylim(-0.37, 0.37)
    plt.xticks([-0.3, 0, 0.3])
    plt.yticks([-0.3, 0, 0.3], rotation=90)
    patch_list = [
        mpatches.Patch(color=color, label=label)
        for color, label in zip(
            [np.array((223, 41, 53, 255))/255, '#2660E8'],
            ['Action', 'Result']
        )
    ]
    plt.legend(handles=patch_list, loc='center')
    plt.show()

# %%
