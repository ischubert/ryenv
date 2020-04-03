"""
Collection of environment classes that are based on rai-python
"""
import sys
import os
import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getenv("HOME") + '/git/rai-python/rai/rai/ry')
import libry as ry

class DiskEnv():
    """
    Wrapper class for the disk-on-a-table environment,
    simulated using PhysX
    """

    def __init__(
            self,
            action_duration=0.5,
            action_length=0.1,
            break_pos_thres=0.03,
            floor_level=0.65,
            finger_relative_level=0.14,
            tau=.01,
            safety_distance=0.1,
            spherically_symmetric_neighbours=False,
            file=None,
            display=False
    ):
        self.action_duration = action_duration
        self.action_length = action_length
        self.break_pos_thres = break_pos_thres
        self.floor_level = floor_level
        self.finger_relative_level = finger_relative_level
        self.tau = tau
        self.safety_distance = safety_distance
        self.spherically_symmetric_neighbours = spherically_symmetric_neighbours

        self.n_steps = int(self.action_duration/self.tau)
        self.proportion_per_step = 1/self.n_steps

        self.config = ry.Config()

        if file is not None:
            self.config.addFile(file)
        else:
            self.config.addFile(os.getenv("HOME") +
                                '/git/ryenv/ryenv/z.push_default.g')

        self.config.makeObjectsFree(['finger'])
        self.config.setJointState([0.3, 0.3, 0.15, 1., 0., 0., 0.])

        self.finger_radius = self.config.frame('finger').info()['size'][0]

        self.simulation = self.config.simulation(
            ry.SimulatorEngine.physx, display)

        self.reset_disk()

        self.disk_dimensions = [0.2, 0.25]
        self.config.frame('box').setShape(
            ry.ST.cylinder, size=self.disk_dimensions)

        self.reset([0.3, 0.3])

    def view(self):
        """
        Create view of current configuration
        """
        return self.config.view()

    def add_and_show_target(self, target_state):
        """
        Add target state and visualize it in view
        """
        target = self.config.addFrame(name="target")
        target.setShape(ry.ST.cylinder, size=self.disk_dimensions)
        target.setColor([1, 1, 0, 0.4])

        self.set_frame_state(
            target_state,
            "target"
        )

    def get_disk_state(self):
        """
        Get the current position of the disk
        """
        return np.array(self.config.frame('box').getPosition()[:2])

    def reset_disk(self, coords=(0, 0)):
        """
        Reset the disk to an arbitrary position
        """
        # always reset box to the center
        self.set_frame_state(
            coords,
            'box'
        )
        state_now = self.config.getFrameState()
        self.simulation.setState(state_now, np.zeros((state_now.shape[0], 6)))

    def allowed_state(
            self,
            finger_position
    ):
        """
        Return whether a state of the finger is within the allowed area or not
        """
        return np.linalg.norm(finger_position) > self.disk_dimensions[0] + self.safety_distance

    def reset(
            self,
            finger_position,
            disk_position=(0, 0)
    ):
        """
        Reset the state (i.e. the finger state) to an arbitrary position
        """
        assert self.allowed_state(finger_position)

        joint_q = np.array([
            *finger_position,
            self.finger_relative_level,
            1.,
            0.,
            0.,
            0.
        ])

        # Monkey patch: When I set the state, the box is initiated at
        # the specified coordinates with velocity 0.
        # However, the finger moves in time tau to its designated spot
        # only if I use simulation.step. This can lead to the finger "kicking
        # away" the box. Thus, I set the joint state, simulate a single step, and set
        # the box state separately
        self.config.setJointState(joint_q)
        self.simulation.step(u_control=[], tau=self.tau)
        self.reset_disk(coords=disk_position)

    def evolve(
            self,
            n_steps=1000,
            fps=None
    ):
        """
        Evolve the simulation for n_steps time steps of length self.tau
        """
        for _ in range(n_steps):
            self.simulation.step(u_control=[], tau=self.tau)
            if fps is not None:
                time.sleep(1/fps)

    def set_frame_state(
            self,
            state,
            frame_name
    ):
        """
        Set an arbitrary frame of the configuration to
        and arbitrary state
        """
        self.config.frame(frame_name).setPosition([
            *state[:2],
            self.floor_level
        ])

    def transition(
            self,
            action,
            fps=None
    ):
        """
        Simulate the system's transition under an action
        """
        # gradual pushing movement
        joint_q = self.config.getJointState()
        for _ in range(self.n_steps):
            joint_q[0] += self.proportion_per_step * action[0]
            joint_q[1] += self.proportion_per_step * action[1]
            self.config.setJointState(joint_q)
            self.simulation.step(u_control=[], tau=self.tau)
            if fps is not None:
                time.sleep(1/fps)

        change = np.array(
            self.config.frame('box').getPosition()[:2]
        )

        return change

    def get_state(self):
        """
        Get the current state, i.e. position of the finger
        """
        return self.config.getJointState()[:2]

    def get_relative_finger_state(self):
        """"
        Get the current state (position of the finger) relative to
        the position of the disk
        """
        disk = self.get_disk_state()
        finger = self.get_state()

        finger_shifted = finger - disk

        return finger_shifted

    def sample_random_goals(self, n_goals):
        """
        This function samples uniformly from the goal distribution
        """
        angle_dir = np.pi*(2*np.random.rand(n_goals)-1)

        return np.stack((
            np.cos(angle_dir),
            np.sin(angle_dir)
        ), axis=-1)

    def calculate_thresholded_change(self, change):
        """Apply threshold to change in order to avoid giving rewards fors numerical noise"""
        change_thresholded = change.copy()
        if np.linalg.norm(change_thresholded) < self.break_pos_thres:
            change_thresholded[0] = 0
            change_thresholded[1] = 0
        return change_thresholded

    def calculate_reward(self, change, goal):
        """calculate reward from intended goal and actual change of disk coordinates"""

        change = self.calculate_thresholded_change(change)

        direction_changed = not sum(change) == 0

        if direction_changed:
            direction_cosine = np.sum(change[:2]*goal[:2])/np.linalg.norm(
                change[:2]
            )/np.linalg.norm(goal[:2])

            if direction_cosine > 0.9:
                return 1
            return -1
        return 0

    def find_near_neighbours(
            self,
            states,
            goals,
            state,
            goal,
            scale
    ):
        """
        This function does a rapneid pre-choice of possible near neighbours only by
        putting constraints on single-coordinate differences on the 5 coordinates
        state_x,state_y,goal_dir_x,goal_dir_y,goal_orientation.
        This greatly reduces the number of pairs the actual distance has to be
        calculated for.
        """
        # only consider samples who have a smaller difference than action_length
        # in all of their state coordinates...
        subset = np.where(
            np.abs(
                states[:, 0] - state[0]
            ) < self.action_length * scale
        )[0]
        subset = subset[
            np.abs(
                states[subset, 1] - state[1]
            ) < self.action_length * scale
        ]

        # ...and who have a smaller difference than 0.1
        # in both of the goal direction coordinates
        subset = subset[
            np.abs(
                goals[subset, 0] - goal[0]
            ) < 0.1 * scale
        ]
        subset = subset[
            np.abs(
                goals[subset, 1] - goal[1]
            ) < 0.1 * scale
        ]

        if self.spherically_symmetric_neighbours:
            # angle-dependent cut-out in goal space
            subset = subset[
                np.sum(
                    goals[subset, :] * goal[None, :],
                    axis=-1
                ) > np.cos(0.1 * scale)
            ]
            # circular cut-out in state-space
            subset = subset[
                np.linalg.norm(
                    states[subset, :] - state[None, :],
                    axis=-1
                ) < self.action_length * scale
            ]

        return subset

    def get_augmented_targets(self, states, targets):
        """
        Create handcrafted targets for the values of some of the states
        """
        targets[
            np.linalg.norm(
                states,
                axis=-1
            ) > 1
        ] = 0

    def visualize_states(self, states, save_name=None):
        """
        Helper function to visualize a collection of states
        """
        plt.plot(
            states[:, 0], states[:, 1], '*'
        )
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        if save_name is not None:
            plt.savefig(save_name + '.png')
        plt.show()

    def test_controller(
            self,
            controller,
            n_of_n_splits=(0, 1),
            n_trial_numbers=20,
            rollout_length=50
    ):
        """
        Create data for a circular plot of the performance of
        the controller in this environment
        """
        direction_angles_all = np.linspace(0, 2*np.pi, 16, endpoint=False)
        direction_angles = np.split(
            direction_angles_all,
            n_of_n_splits[1]
            )[n_of_n_splits[0]]

        all_rewards = []

        for direction_angle in tqdm.tqdm(direction_angles):
            goal = np.array([
                np.cos(direction_angle),
                np.sin(direction_angle)
            ])
            rewards = []

            trial_number = 0
            while trial_number < n_trial_numbers:
                possible_finger_state = 1*(np.random.rand(2)-0.5)
                if self.allowed_state(possible_finger_state) and (
                        sum(goal*possible_finger_state)/np.linalg.norm(
                            goal)/np.linalg.norm(possible_finger_state) < 0
                ):
                    trial_number += 1
                    self.reset(
                        possible_finger_state,
                        disk_position=[0, 0]
                    )

                    for __ in range(rollout_length):
                        action = controller.get_action(
                            self.get_state(), goal
                        )

                        # TODO das hier muss der controller machen, das env weiss nix von all dem
                        # action = controller.local_dynamics_model.action_length \
                        #     * action/np.linalg.norm(action)

                        if any(np.isnan(action)):
                            raise Exception('action is nan')

                        change = self.transition(action)

                        if np.sum(np.abs(
                                self.calculate_thresholded_change(change)
                        )) != 0:
                            break

                    if np.sum(np.abs(
                            self.calculate_thresholded_change(change)
                    )) == 0:
                        reward = -10
                    else:
                        reward = np.sum(np.array(change)*np.array(
                            goal))/np.linalg.norm(change)/np.linalg.norm(goal)

                    print(goal, self.calculate_thresholded_change(change), reward)
                    rewards.append(reward)

            all_rewards.append(rewards)
        return all_rewards
