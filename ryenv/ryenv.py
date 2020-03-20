import sys
import os
import time
import numpy as np

sys.path.append(os.getenv("HOME") + '/git/rai-python/rai/rai/ry')
import libry as ry

class disk_env():
    def __init__(
        self,
        action_duration=0.5,
        floor_level=0.65,
        finger_relative_level=0.14,
        tau=.01,
        safety_distance=0.005,
        file=None,
        display=False
    ):
        self.action_duration = action_duration
        self.floor_level = floor_level
        self.finger_relative_level = finger_relative_level
        self.tau = tau
        self.safety_distance = safety_distance

        self.n_steps = int(self.action_duration/self.tau)
        self.proportion_per_step = 1/self.n_steps

        self.C = ry.Config()

        if file is not None:
            self.C.addFile(file)
        else:
            self.C.addFile(os.getenv("HOME") + '/git/ryenv/ryenv/z.push_default.g')

        self.C.makeObjectsFree(['finger'])
        self.C.setJointState([0.3,0.3,0.15,1,0,0,0])

        self.finger_radius = self.C.frame('finger').info()['size'][0]

        self.S = self.C.simulation(ry.SimulatorEngine.physx, display)

        self.reset_disk()
        self.Xstart = self.C.getFrameState().copy()

        self.disk_dimensions = [0.2, 0.25]
        self.C.frame('box').setShape(ry.ST.cylinder, size=self.disk_dimensions)

        self.reset([0.3,0.3])

    def view(self):
        return self.C.view()
    
    def add_and_show_target(self,target_state):
        target = self.C.addFrame(name="target")
        target.setShape(ry.ST.cylinder, size=self.disk_dimensions)
        target.setColor([1,1,0,0.4])
        
        self.set_frame_state(
            target_state,
            "target"
        )
    
    def get_disk_state(self):
        return np.array(self.C.frame('box').getPosition()[:2])
    
    def reset_disk(self,coords = (0,0)):
        # always reset box to the center
        self.set_frame_state(
            coords,
            'box'
        )
        state_now = self.C.getFrameState()
        self.S.setState(state_now, np.zeros((state_now.shape[0],6)))
    
    def finger_position_outside_of_starting_disk(
        self,
        finger_position
    ):
        return np.linalg.norm(finger_position) > self.disk_dimensions[0] + 0.06
        
    def reset(
        self,
        finger_position,
        disk_position = (0,0)
    ):  
        assert self.finger_position_outside_of_starting_disk(finger_position)
        
        q = np.array([
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
        # only if I use S.step. This can lead to the finger "kicking
        # away" the box. Thus, I set the joint state, simulate a single step, and set
        # the box state separately
        self.C.setJointState(q)
        self.S.step(u_control = [],tau = self.tau)
        self.reset_disk(coords=disk_position)
        
    def evolve(
        self,
        n_steps = 1000,
        fps = None
    ):
        for _ in range(n_steps):
            self.S.step(u_control = [],tau = self.tau)
            if fps is not None:
                time.sleep(1/fps)
        
    def set_frame_state(
        self,
        state,
        frame_name
    ):
        self.C.frame(frame_name).setPosition([
            *state[:2],
            self.floor_level
        ])
    
    def transition(
        self,
        action,
        fps = None
    ):
        # gradual pushing movement
        q = self.C.getJointState()
        for _ in range(self.n_steps):
            q[0] += self.proportion_per_step * action[0]
            q[1] += self.proportion_per_step * action[1]
            self.C.setJointState(q)
            self.S.step(u_control = [],tau = self.tau)
            if fps is not None:
                time.sleep(1/fps)
        
        change = np.array(
            self.C.frame('box').getPosition()[:2]
        )
        
        return change
    
    def transition_transformed(
        self,
        action,
        fps = None
    ):
        box = self.get_disk_state()
        return self.transition(
            np.matmul(
                np.array([
                    [np.cos(-box[-1]),np.sin(-box[-1])],
                    [-np.sin(-box[-1]),np.cos(-box[-1])]
                ]),
                action
            ),
            fps=fps
        )
    
    def get_finger_state(self):
        return self.C.getJointState()[:2]
    
    def get_relative_finger_state(self):
        box = self.get_disk_state()
        finger = self.get_finger_state()
        
        finger_shifted = finger - box[:2]
        finger_transformed = np.matmul(
            np.array([
                [np.cos(box[-1]),np.sin(box[-1])],
                [-np.sin(box[-1]),np.cos(box[-1])]
            ]),
            finger_shifted
        )
        
        return finger_transformed