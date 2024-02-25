import sys
sys.path.insert(0, '/home/federico/GitHub/safe-control-gym')

import numpy as np

from safe_control_gym.controllers.firmware.firmware_wrapper import Command


class Controller():
    '''Template controller class. '''

    def __init__(self,
                 initial_obs,
                 initial_info,
                 ):
        '''Initialization of the controller.

        Args:
            initial_obs (ndarray): The initial observation of the quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            initial_info (dict): The a priori information as a dictionary.
        '''
        # Save environment and conrol parameters.
        self.CTRL_FREQ = initial_info['ctrl_freq']
        self.CTRL_DT = 1.0 / self.CTRL_FREQ
        self.initial_obs = initial_obs

        self.prev_roll = 0
        self.prev_pitch = 0

        self.traj = np.concatenate((np.linspace(0.4, 0.4, 15), np.linspace(-0.4, -0.4, 30), np.linspace(0.4, 0.4, 30), np.linspace(-0.4, -0.4, 25)))

        self.static_RP = []
        self.bias_roll = 0.0
        self.bias_pitch = 0.0

        # Reset counters and buffers.
        self.reset()

    def cmdFirmware(self,
                    time,
                    obs,
                    ):
        '''Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].

        Returns:
            Command: selected type of command (takeOff, cmdFullState, etc., see Enum-like class `Command`).
            List: arguments for the type of command (see comments in class `Command`)
        '''

        iteration = int(time * self.CTRL_FREQ)

        obs[6] -= self.bias_roll
        obs[7] -= self.bias_pitch

        if abs(obs[6]) > 0.78:
            obs[6] = self.prev_roll
        if abs(obs[7]) > 0.78:
            obs[7] = self.prev_pitch

        self.prev_roll = obs[6]
        self.prev_pitch = obs[7]

        if iteration < self.CTRL_FREQ:
            self.static_RP.append([obs[6], obs[7]])
            command_type = Command(0)  # None.
            args = []
        elif iteration == self.CTRL_FREQ:
            self.bias_roll = np.mean([angle[0] for angle in self.static_RP])
            self.bias_pitch = np.mean([angle[1] for angle in self.static_RP])
            print(f'Biased Roll: {self.bias_roll}, Biased Pitch: {self.bias_pitch}')
            print(f'Iter: {iteration} - Take off.')
            height = 1
            duration = 2

            command_type = Command(2)  # Take-off.
            args = [height, duration]
        elif iteration >= 4 * self.CTRL_FREQ and iteration < 8 * self.CTRL_FREQ:
            des_roll = self.traj[iteration - 4*self.CTRL_FREQ]
            des_pitch = 0
            print(f'Iter: {iteration} - cmdVel.')
            command_type = Command(7)  # Try sending cmdVel
            print(f'Roll: {obs[6]}, Pitch: {obs[7]}, Des Roll: {des_roll}')
            args = [des_roll*180.0/np.pi, des_pitch, 0, 0]
        elif iteration == 8 * self.CTRL_FREQ:
            print(f'Iter: {iteration} - NOTIFYSETPOINTSTOP.')
            command_type = Command(6)  # Try sending NOTIFYSETPOINTSTOP
            args = []
        elif iteration == 8 * self.CTRL_FREQ + 1:
            print(f'Iter: {iteration} - Landing.')
            height = 0.2
            duration = 3

            command_type = Command(3)  # Land.
            args = [height, duration]
        elif iteration == 11 * self.CTRL_FREQ + 1:
            print(f'Iter: {iteration} - Terminating.')
            command_type = Command(-1)  # Terminate.
            args = []
        else:
            print(f'Iter: {iteration}')
            command_type = Command(0)  # None.
            args = []

        return command_type, args

    def reset(self):
        '''Reset. '''
        pass
