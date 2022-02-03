"""A quadrotor trajectory tracking example.

Notes:
    Includes and uses PID control.

Run as:

    $ python3 ./pid_experiment.py --task quadrotor --algo pid --overrides ./config_pid_quadrotor.yaml

"""
import time
import pybullet as p
from functools import partial

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

def main():
    """The main function creating, running, and closing an environment.

    """

    # Create an environment
    CONFIG_FACTORY = ConfigFactory()               
    config = CONFIG_FACTORY.merge()
    
    # Set iterations and episode counter.
    ITERATIONS = int(config.quadrotor_config['episode_len_sec']*config.quadrotor_config['ctrl_freq'])
    
    # Start a timer.
    START = time.time()
    
    # Create controller.
    env_func = partial(make,
                    config.task,
                    **config.quadrotor_config
                    )
    ctrl = make(config.algo,
                env_func,
                )
                
    ctrl.reset()

    initial_info = ctrl.initial_info

    # Plot trajectory.
    for i in range(0, initial_info['x_reference'].shape[0], 10):
        p.addUserDebugLine(lineFromXYZ=[initial_info['x_reference'][i-10,0], 0, initial_info['x_reference'][i-10,2]],
                            lineToXYZ=[initial_info['x_reference'][i,0], 0, initial_info['x_reference'][i,2]],
                            lineColorRGB=[1, 0, 0],
                            physicsClientId=ctrl.env.PYB_CLIENT)

    # Run the experiment.
    results = ctrl.run( iterations=ITERATIONS)
            
    # Plot the experiment.
    for i in range(ITERATIONS):
        # Step the environment and print all returned information.
        obs, reward, done, info, action = results['obs'][i], results['reward'][i], results['done'][i], results['info'][i], results['action'][i]
        
        # Print the last action and the information returned at each step.
        print(i, '-th step.')
        print(action, '\n', obs, '\n', reward, '\n', done, '\n', info, '\n')
        
        if done:
            ctrl.close()

    elapsed_sec = time.time() - START
    print("\n{:d} iterations (@{:d}Hz) and {:d} episodes in {:.2f} seconds, i.e. {:.2f} steps/sec for a {:.2f}x speedup.\n"
            .format(ITERATIONS, config.quadrotor_config.ctrl_freq, 1, elapsed_sec, ITERATIONS/elapsed_sec, (ITERATIONS*(1. / config.quadrotor_config.ctrl_freq))/elapsed_sec))


if __name__ == "__main__":
    main()