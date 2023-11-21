import sys
sys.path.insert(0, '/home/federico/GitHub/safe-control-gym')

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

        if iteration == 0:
            print(f'Iter: {iteration} - Take off.')
            height = 1
            duration = 2

            command_type = Command(2)  # Take-off.
            args = [height, duration]
        elif iteration == 3 * self.CTRL_FREQ:
            print(f'Iter: {iteration} - Landing.')
            height = 0.1
            duration = 3

            command_type = Command(3)  # Land.
            args = [height, duration]
        elif iteration == 6 * self.CTRL_FREQ + 1:
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
