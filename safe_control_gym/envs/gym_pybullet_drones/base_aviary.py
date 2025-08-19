'''BaseAviary module.

This module contains the superclass of the Quadrotor environment, BaseAviary.
BaseAviary implements most of the integration with PyBullet.

The module also contains enumerations for drone models, PyBullet physics updates
image types captured by PyBullet's camera rendering.
'''

import os
import pkgutil
import time
import xml.etree.ElementTree as etxml
from datetime import datetime
from enum import Enum

import casadi as cs
import numpy as np
import pybullet as p
import pybullet_data

from safe_control_gym.envs.benchmark_env import BenchmarkEnv
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import ThrustTransformer
from safe_control_gym.math_and_models.transformations import csRotXYZ, get_angularvelocity_rpy

egl = pkgutil.get_loader('eglRenderer')


class DroneModel(str, Enum):
    '''Drone models enumeration class.'''

    CF2X = 'cf2x'  # Bitcraze Craziflie 2.0 in the X configuration.


class Physics(str, Enum):
    '''Physics implementations enumeration class.'''

    PYB = 'pyb'  # Base PyBullet physics update.
    DYN = 'dyn'  # Update with an explicit model of the dynamics.
    PYB_GND = 'pyb_gnd'  # PyBullet physics update with ground effect.
    PYB_DRAG = 'pyb_drag'  # PyBullet physics update with drag.
    PYB_DW = 'pyb_dw'  # PyBullet physics update with downwash.
    PYB_GND_DRAG_DW = 'pyb_gnd_drag_dw'  # PyBullet physics update with ground effect, drag, and downwash.
    RK4 = 'rk4'  # Update with a Runge-Kutta 4th order integrator.
    DYN_2D = 'dyn_2d'  # Update with Physics 5 state dynamics model
    DYN_SI = 'dyn_si'  # Update with SysId 6 state dynamics model
    DYN_SI_3D = 'dyn_si_3d'  # Update with SysId 12 state dynamics model
    DYN_SI_3D_10 = 'dyn_si_3d_10'  # Update with SysId 10 state dynamics model
    DYN_SI_3D_DELAY = 'dyn_si_3d_delay'  # Update with SysId 12 state dynamics model with delay


class ImageType(int, Enum):
    '''Camera capture image type enumeration class.'''

    RGB = 0  # Red, green, blue (and alpha).
    DEP = 1  # Depth.
    SEG = 2  # Segmentation by object id.
    BW = 3  # Black and white.


class BaseAviary(BenchmarkEnv):
    '''Base class for 'drone aviary' Gym environments.'''
    NAME = 'base_aviary'
    URDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 physics: Physics = Physics.PYB,
                 record=False,
                 gui=False,
                 verbose=False,
                 **kwargs):
        '''Initialization of a generic aviary environment.

        Args:
            drone_model (DroneModel, optional): The desired drone type (detailed in an .urdf file
                                                in folder `assets`).
            num_drones (int, optional): The desired number of drones in the aviary.
            physics (Physics, optional): The desired implementation of PyBullet physics/custom
                                         dynamics.
            record (bool, optional): Whether to save a video of the simulation in folder
                                     `files/videos/`.
            gui (bool, optional): Whether to use PyBullet's GUI.
            verbose (bool, optional): If to suppress environment print statements.
        '''
        # Constants.
        self.GRAVITY_ACC = 9.8
        self.RAD2DEG = 180 / np.pi
        self.DEG2RAD = np.pi / 180
        # Parameters.
        self.DRONE_MODEL = DroneModel(drone_model)
        self.URDF_PATH = os.path.join(self.URDF_DIR, self.DRONE_MODEL.value + '.urdf')
        self.NUM_DRONES = num_drones
        self.PHYSICS = Physics(physics)
        self.RECORD = record
        # Load the drone properties from the .urdf file.
        self.MASS, \
            self.L, \
            self.THRUST2WEIGHT_RATIO, \
            self.J, \
            self.J_INV, \
            self.KF, \
            self.KM, \
            self.COLLISION_H, \
            self.COLLISION_R, \
            self.COLLISION_Z_OFFSET, \
            self.MAX_SPEED_KMH, \
            self.GND_EFF_COEFF, \
            self.PROP_RADIUS, \
            self.DRAG_COEFF, \
            self.DW_COEFF_1, \
            self.DW_COEFF_2, \
            self.DW_COEFF_3, \
            self.PWM2RPM_SCALE, \
            self.PWM2RPM_CONST, \
            self.MIN_PWM, \
            self.MAX_PWM = self._parse_urdf_parameters(self.URDF_PATH)
        self.GROUND_PLANE_Z = -0.05
        if verbose:
            print(
                '[INFO] BaseAviary.__init__() loaded parameters from the drone\'s .urdf: \
                \n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f}, \
                \n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f}, \
                \n[INFO] gnd_eff_coeff {:f}, prop_radius {:f}, \
                \n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f}, \
                \n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f} \
                \n[INFO] pwm2rpm_scale {:f}, pwm2rpm_const {:f}, min_pwm {:f}, max_pwm {:f}'
                .format(self.MASS, self.L, self.J[0, 0], self.J[1, 1], self.J[2, 2],
                        self.KF, self.KM, self.THRUST2WEIGHT_RATIO,
                        self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS,
                        self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1,
                        self.DW_COEFF_2, self.DW_COEFF_3, self.PWM2RPM_SCALE,
                        self.PWM2RPM_CONST, self.MIN_PWM, self.MAX_PWM))
        # Compute constants.
        self.GRAVITY = self.GRAVITY_ACC * self.MASS
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4 * self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO * self.GRAVITY) / (4 * self.KF))
        self.MAX_THRUST = (4 * self.KF * self.MAX_RPM ** 2)
        self.MAX_XY_TORQUE = (self.L * self.KF * self.MAX_RPM ** 2)
        self.MAX_Z_TORQUE = (2 * self.KM * self.MAX_RPM ** 2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt(
            (15 * self.MAX_RPM ** 2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)
        # BenchmarkEnv constructor.
        super().__init__(gui=gui, verbose=verbose, **kwargs)
        # Connect to PyBullet.
        self.PYB_CLIENT = -1
        if gui:
            # With debug GUI.
            self.PYB_CLIENT = p.connect(p.GUI)  # p.connect(p.GUI, options='--opengl2')
            p.resetDebugVisualizerCamera(cameraDistance=3,
                                         cameraYaw=-30,
                                         cameraPitch=-30,
                                         cameraTargetPosition=[0, 0, 0],
                                         physicsClientId=self.PYB_CLIENT)
            ret = p.getDebugVisualizerCamera(physicsClientId=self.PYB_CLIENT)
            if verbose:
                print('viewMatrix', ret[2])
                print('projectionMatrix', ret[3])
        else:
            # Without debug GUI.
            self.PYB_CLIENT = p.connect(p.DIRECT)
            # Uncomment the following line to use EGL Render Plugin #
            # Instead of TinyRender (CPU-based) in PYB's Direct mode
            # if platform == 'linux':
            #     p.setAdditionalSearchPath(pybullet_data.getDataPath())
            #     plugin = p.loadPlugin(egl.get_filename(), '_eglRendererPlugin')
            #     print('plugin=', plugin)
        self.RENDER_WIDTH = int(640)
        self.RENDER_HEIGHT = int(480)
        self.FRAME_PER_SEC = 24
        self.CAPTURE_FREQ = int(self.PYB_FREQ / self.FRAME_PER_SEC)
        self.CAM_VIEW = p.computeViewMatrixFromYawPitchRoll(
            distance=3,
            yaw=-30,
            pitch=-30,
            roll=0,
            cameraTargetPosition=[0, 0, 0],
            upAxisIndex=2,
            physicsClientId=self.PYB_CLIENT)
        self.CAM_PRO = p.computeProjectionMatrixFOV(fov=60.0,
                                                    aspect=self.RENDER_WIDTH / self.RENDER_HEIGHT,
                                                    nearVal=0.1,
                                                    farVal=1000.0)
        # Set default initial poses when loading drone's urdf model.
        # can be overriden later for specific tasks (as sub-classes) in reset()
        self.INIT_XYZS = np.vstack([np.array([x * 4 * self.L for x in range(self.NUM_DRONES)]),
                                    np.array([y * 4 * self.L for y in range(self.NUM_DRONES)]),
                                    np.ones(self.NUM_DRONES) * (self.COLLISION_H / 2 - self.COLLISION_Z_OFFSET)
                                    ]).transpose().reshape(self.NUM_DRONES, 3)
        self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        if physics == Physics.RK4:
            self.setup_rk4_dynamics_expression()
        elif physics == Physics.DYN_2D:
            self.setup_dynamics_2d_expression()
        elif physics == Physics.DYN_SI:
            self.setup_dynamics_si_expression()
        elif physics == Physics.DYN_SI_3D:
            self.setup_dynamics_si_3d_expression()
        elif physics == Physics.DYN_SI_3D_10:
            self.setup_dynamics_si_3d_10_expression()
        elif physics == Physics.DYN_SI_3D_DELAY:
            self.setup_thrust_transformation()
            self.setup_dynamics_si_3d_delay_expression()

    def setup_thrust_transformation(self):
        '''Set up the thrust transformation parameters.

        This method sets up the transformation parameters for converting
        thrust to RPM and vice versa.
        '''
        self.transform_params = {
            'f_min': -1,
            'f_max': 1,
            'cmd_min': -1,
            'cmd_max': 1,
            # 0.2012 0.4553 0.2389 0.4750
            # 'f_min': 0.2389,
            # 'f_max': 0.4750,
            # 'cmd_min': 0.2012,
            # 'cmd_max': 0.4553,
            'f_hover': self.MASS * self.GRAVITY_ACC,
            'cmd_hover': self.MASS * self.GRAVITY_ACC,
            'transformation_mode': 3  # 1: no transform, 2: subtract hover, 3: normalize
        }
        self.thrust_transform = ThrustTransformer(transform_params=self.transform_params)

    def close(self):
        '''Terminates the environment.'''
        if self.RECORD and self.GUI:
            p.stopStateLogging(self.VIDEO_ID, physicsClientId=self.PYB_CLIENT)
        if self.PYB_CLIENT >= 0:
            p.disconnect(physicsClientId=self.PYB_CLIENT)
        self.PYB_CLIENT = -1

    def _reset_simulation(self):
        '''Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.
        '''
        # Initialize/reset counters and zero-valued variables.
        self.RESET_TIME = time.time()
        self.first_render_call = True
        self.X_AX = -1 * np.ones(self.NUM_DRONES)
        self.Y_AX = -1 * np.ones(self.NUM_DRONES)
        self.Z_AX = -1 * np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1 * np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM = False
        self.last_input_switch = 0
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))
        self.gui_input = np.zeros(4)
        # Initialize the drones kinematic information.
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        self.rpy_rates = np.zeros((self.NUM_DRONES, 3))
        self.motor_forces = np.zeros((self.NUM_DRONES, 1))
        if self.PHYSICS in [Physics.DYN_SI_3D_DELAY] \
                and hasattr(self, 'init_tau'):
            self.motor_forces = np.ones((self.NUM_DRONES, 1)) * self.init_tau
        # if (self.PHYSICS == Physics.DYN or self.PHYSICS == Physics.RK4
        #         or self.PHYSICS == Physics.DYN_2D or self.PHYSICS == Physics.DYN_SI):
        #     self.rpy_rates = np.zeros((self.NUM_DRONES, 3))

        # Set PyBullet's parameters.
        p.resetSimulation(physicsClientId=self.PYB_CLIENT)
        p.setGravity(0, 0, -self.GRAVITY_ACC, physicsClientId=self.PYB_CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.PYB_CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.PYB_CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.PYB_CLIENT)
        # Load ground plane, drone and obstacles models.
        self.PLANE_ID = p.loadURDF('plane.urdf', [0, 0, self.GROUND_PLANE_Z],
                                   physicsClientId=self.PYB_CLIENT)
        self.DRONE_IDS = np.array([
            p.loadURDF(self.URDF_PATH,
                       self.INIT_XYZS[i, :],
                       p.getQuaternionFromEuler(self.INIT_RPYS[i, :]),
                       flags=p.URDF_USE_INERTIA_FROM_FILE,
                       physicsClientId=self.PYB_CLIENT)
            for i in range(self.NUM_DRONES)
        ])
        for i in range(self.NUM_DRONES):
            p.changeDynamics(self.DRONE_IDS[i], -1, linearDamping=0, angularDamping=0)
        # Update and store the drones kinematic information.
        self._update_and_store_kinematic_information()
        # Start video recording.
        self._start_video_recording()
        # # Show frame of references of drones, will severely slow down the GUI.
        # for i in range(self.NUM_DRONES):
        # if gui:
        #     self._show_drone_local_axes(i)

    def _advance_simulation(self, clipped_action, disturbance_force=None):
        '''Advances the environment by one simulation step.

        Args:
            clipped_action (ndarray): The input action for one or more drones,
                                         as RPMs by the specific implementation of
                                         `_preprocess_action()` in each subclass.
            disturbance_force (ndarray, optional): Disturbance force, applied to all drones.
        '''
        # clipped_action = np.reshape(clipped_action, (self.NUM_DRONES, 4))
        clipped_action = np.expand_dims(clipped_action, axis=0)

        # Repeat for as many as the aggregate physics steps.
        for _ in range(self.PYB_STEPS_PER_CTRL):
            # Update and store the drones kinematic info for certain
            # Between aggregate steps for certain types of update.
            if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [
                Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG,
                Physics.PYB_DW, Physics.PYB_GND_DRAG_DW, Physics.RK4  # , Physics.DYN_2D
            ]:
                self._update_and_store_kinematic_information()
            # Step the simulation using the desired physics update.
            for i in range(self.NUM_DRONES):
                executable_action = self._preprocess_control(clipped_action[i, :])
                if self.PHYSICS == Physics.PYB:
                    self._physics(executable_action, i)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.DYN_2D:
                    self._dynamics_2d(executable_action, i)
                elif self.PHYSICS == Physics.DYN_SI:
                    self._dynamics_si(executable_action, i, disturbance_force)
                elif self.PHYSICS == Physics.RK4:
                    self._dynamics_rk4(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.DYN_SI_3D:
                    self._dynamics_si_3d(executable_action, i, disturbance_force)
                elif self.PHYSICS == Physics.DYN_SI_3D_10:
                    self._dynamics_si_3d_10(executable_action, i, disturbance_force)
                elif self.PHYSICS == Physics.DYN_SI_3D_DELAY:
                    self._dynamics_si_3d_delay(executable_action, i, disturbance_force)
                elif self.PHYSICS == Physics.PYB_GND:
                    self._physics(clipped_action[i, :], i)
                    self._ground_effect(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DRAG:
                    self._physics(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DW:
                    self._physics(clipped_action[i, :], i)
                    self._downwash(i)
                elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[i, :], i)
                    self._ground_effect(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                    self._downwash(i)
                # Apply disturbance
                if disturbance_force is not None:
                    pos = self._get_drone_state_vector(i)[:3]
                    '''
                    NOTE: applyExternalForce only works when explicitly
                    stepping the simulation with p.stepSimulation().
                    Therefore,
                    '''
                    p.applyExternalForce(
                        self.DRONE_IDS[i],
                        linkIndex=4,  # Link attached to the quadrotor's center of mass.
                        forceObj=disturbance_force,
                        posObj=pos,
                        flags=p.WORLD_FRAME,
                        physicsClientId=self.PYB_CLIENT)
            # PyBullet computes the new state, unless Physics.DYN.
            if self.PHYSICS not in [Physics.DYN, Physics.RK4, Physics.DYN_2D, Physics.DYN_SI,
                                    Physics.DYN_SI_3D, Physics.DYN_SI_3D_10, Physics.DYN_SI_3D_DELAY]:
                p.stepSimulation(physicsClientId=self.PYB_CLIENT)
            # Save the last applied action (e.g. to compute drag).
            self.last_clipped_action = clipped_action
        if self.PHYSICS in [Physics.DYN_2D, Physics.DYN_SI,
                            Physics.DYN_SI_3D, Physics.DYN_SI_3D_10,
                            Physics.DYN_SI_3D_DELAY]:
            # set the state of the drone after stepping with the analytical model
            self._set_pybullet_information()
        # Update and store the drones kinematic information.
        self._update_and_store_kinematic_information()

    def render(self, mode='human', close=False):
        '''Prints a textual output of the environment.

        Args:
            mode (str, optional): Unused.
            close (bool, optional): Unused.
        '''
        if self.first_render_call and not self.GUI:
            print(
                '[WARNING] BaseAviary.render() is implemented as text-only, re-initialize the environment using '
                'Aviary(gui=True) to use PyBullet\'s graphical interface'
            )
            self.first_render_call = False
        if self.VERBOSE:
            print(
                '\n[INFO] BaseAviary.render() ——— it {:04d}'.format(
                    self.pyb_step_counter),
                '——— wall-clock time {:.1f}s,'.format(time.time() - self.RESET_TIME),
                'simulation time {:.1f}s@{:d}Hz ({:.2f}x)'.format(
                    self.pyb_step_counter * self.PYB_TIMESTEP, self.SIM_FREQ,
                    (self.pyb_step_counter * self.PYB_TIMESTEP) / (time.time() - self.RESET_TIME)))
            for i in range(self.NUM_DRONES):
                print(
                    '[INFO] BaseAviary.render() ——— drone {:d}'.format(i),
                    '——— x {:+06.2f}, y {:+06.2f}, z {:+06.2f}'.format(
                        self.pos[i, 0], self.pos[i, 1], self.pos[i, 2]),
                    '——— velocity {:+06.2f}, {:+06.2f}, {:+06.2f}'.format(
                        self.vel[i, 0], self.vel[i, 1], self.vel[i, 2]),
                    '——— roll {:+06.2f}, pitch {:+06.2f}, yaw {:+06.2f}'.format(
                        self.rpy[i, 0] * self.RAD2DEG,
                        self.rpy[i, 1] * self.RAD2DEG,
                        self.rpy[i, 2] * self.RAD2DEG),
                    '——— angular velocity {:+06.4f}, {:+06.4f}, {:+06.4f} ——— '.
                    format(self.ang_v[i, 0], self.ang_v[i, 1], self.ang_v[i,
                                                                          2]))

    def _set_pybullet_information(self):
        """Set pybullet state information from external simulation"""
        for i in range(self.NUM_DRONES):
            # Set PyBullet's state.
            p.resetBasePositionAndOrientation(self.DRONE_IDS[i],
                                              self.pos[i, :],
                                              p.getQuaternionFromEuler(self.rpy[i, :]),
                                              physicsClientId=self.PYB_CLIENT)
            # Note: the base's velocity only stored and not used #
            p.resetBaseVelocity(self.DRONE_IDS[i],
                                self.vel[i, :],
                                self.ang_v[i, :],  # ang_vel not computed by DYN
                                physicsClientId=self.PYB_CLIENT)

    def _update_and_store_kinematic_information(self):
        '''Updates and stores the drones kinematic information.

        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).
        '''
        for i in range(self.NUM_DRONES):
            self.pos[i], self.quat[i] = p.getBasePositionAndOrientation(
                self.DRONE_IDS[i], physicsClientId=self.PYB_CLIENT)
            self.rpy[i] = p.getEulerFromQuaternion(self.quat[i])
            self.vel[i], self.ang_v[i] = p.getBaseVelocity(
                self.DRONE_IDS[i], physicsClientId=self.PYB_CLIENT)

    def _start_video_recording(self):
        '''Starts the recording of a video output.

        The format of the video output is .mp4, if GUI is True, or .png, otherwise.
        The video is saved under folder `files/videos`.
        '''
        if self.RECORD and self.GUI:
            self.VIDEO_ID = p.startStateLogging(
                loggingType=p.STATE_LOGGING_VIDEO_MP4,
                fileName=os.path.join(self.output_dir,
                                      'videos/video-{}.mp4'.format(datetime.now().strftime('%m.%d.%Y_%H.%M.%S'))),
                physicsClientId=self.PYB_CLIENT)
        if self.RECORD and not self.GUI:
            self.FRAME_NUM = 0
            self.IMG_PATH = os.path.join(self.output_dir, 'quadrotor_videos/video-{}/'.format(
                datetime.now().strftime('%m.%d.%Y_%H.%M.%S')))
            os.makedirs(os.path.dirname(self.IMG_PATH), exist_ok=True)

    def _get_drone_state_vector(self, nth_drone):
        '''Returns the state vector of the n-th drone.

        Args:
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.

        Returns:
            ndarray. (20, )-shaped array of floats containing the state vector of the n-th drone.
                     Check the only line in this method and `_update_and_store_kinematic_information()`
                     to understand its format.
        '''
        state = np.hstack([
            self.pos[nth_drone, :], self.quat[nth_drone, :],
            self.rpy[nth_drone, :], self.vel[nth_drone, :],
            self.ang_v[nth_drone, :], self.rpy_rates[nth_drone, :], self.motor_forces[nth_drone, :]
        ])
        # state.reshape(20, )
        return state.copy()

    def _physics(self, rpm, nth_drone):
        '''Base PyBullet physics implementation.

        Args:
            rpm (ndarray): (4)-shaped array of ints containing the RPMs values of the 4 motors.
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.
        '''
        forces = np.array(rpm ** 2) * self.KF
        torques = np.array(rpm ** 2) * self.KM
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):
            p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                 i,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.PYB_CLIENT)
        p.applyExternalTorque(self.DRONE_IDS[nth_drone],
                              4,
                              torqueObj=[0, 0, z_torque],
                              flags=p.LINK_FRAME,
                              physicsClientId=self.PYB_CLIENT)

    def _ground_effect(self, rpm, nth_drone):
        '''PyBullet implementation of a ground effect model.

        Inspired by the analytical model used for comparison in (Shi et al., 2019).

        Args:
            rpm (ndarray): (4)-shaped array of ints containing the RPMs values of the 4 motors.
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.
        '''
        # Kin. info of all links (propellers and center of mass)
        link_states = np.array(
            p.getLinkStates(self.DRONE_IDS[nth_drone],
                            linkIndices=[0, 1, 2, 3, 4],
                            computeLinkVelocity=1,
                            computeForwardKinematics=1,
                            physicsClientId=self.PYB_CLIENT))
        # Simple, per-propeller ground effects.
        prop_heights = np.array([
            link_states[0, 0][2], link_states[1, 0][2], link_states[2, 0][2],
            link_states[3, 0][2]
        ])
        prop_heights = np.clip(prop_heights, self.GND_EFF_H_CLIP, np.inf)
        gnd_effects = np.array(rpm ** 2) * self.KF * self.GND_EFF_COEFF \
            * (self.PROP_RADIUS / (4 * prop_heights)) ** 2
        if np.abs(self.rpy[nth_drone, 0]) < np.pi / 2 and np.abs(
                self.rpy[nth_drone, 1]) < np.pi / 2:
            for i in range(4):
                p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                     i,
                                     forceObj=[0, 0, gnd_effects[i]],
                                     posObj=[0, 0, 0],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.PYB_CLIENT)
        # TODO: a more realistic model accounting for the drone's
        # Attitude and its z-axis velocity in the world frame.

    def _drag(self, rpm, nth_drone):
        '''PyBullet implementation of a drag model.

        Based on the the system identification in (Forster, 2015).

        Args:
            rpm (ndarray): (4)-shaped array of ints containing the RPMs values of the 4 motors.
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.
        '''
        # Rotation matrix of the base.
        base_rot = np.array(p.getMatrixFromQuaternion(
            self.quat[nth_drone, :])).reshape(3, 3)
        # Simple draft model applied to the base/center of mass #
        drag_factors = -1 * self.DRAG_COEFF * np.sum(
            np.array(2 * np.pi * rpm / 60))
        drag = np.dot(base_rot, drag_factors * np.array(self.vel[nth_drone, :]))
        p.applyExternalForce(self.DRONE_IDS[nth_drone],
                             4,
                             forceObj=drag,
                             posObj=[0, 0, 0],
                             flags=p.LINK_FRAME,
                             physicsClientId=self.PYB_CLIENT)

    def _downwash(self, nth_drone):
        '''PyBullet implementation of a ground effect model.

        Based on experiments conducted at the Dynamic Systems Lab by SiQi Zhou.

        Args:
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.
        '''
        for i in range(self.NUM_DRONES):
            delta_z = self.pos[i, 2] - self.pos[nth_drone, 2]
            delta_xy = np.linalg.norm(
                np.array(self.pos[i, 0:2]) - np.array(self.pos[nth_drone, 0:2]))
            if delta_z > 0 and delta_xy < 10:  # Ignore drones more than 10 meters away
                alpha = self.DW_COEFF_1 * (self.PROP_RADIUS / (4 * delta_z)) ** 2
                beta = self.DW_COEFF_2 * delta_z + self.DW_COEFF_3
                downwash = [0, 0, -alpha * np.exp(-.5 * (delta_xy / beta) ** 2)]
                p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                     4,
                                     forceObj=downwash,
                                     posObj=[0, 0, 0],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.PYB_CLIENT)

    def _dynamics(self, rpm, nth_drone):
        '''Explicit dynamics implementation.

        Based on code written at the Dynamic Systems Lab by James Xu.

        Args:
            rpm (ndarray): (4)-shaped array of ints containing the RPMs values of the 4 motors.
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.
        '''
        # Current state.
        pos = self.pos[nth_drone, :]
        quat = self.quat[nth_drone, :]
        rpy = self.rpy[nth_drone, :]
        vel = self.vel[nth_drone, :]
        rpy_rates = self.rpy_rates[nth_drone, :]
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        # Compute forces and torques.
        forces = np.array(rpm ** 2) * self.KF
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = np.dot(rotation, thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, self.GRAVITY])
        z_torques = np.array(rpm ** 2) * self.KM
        z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
        if self.DRONE_MODEL == DroneModel.CF2X:
            x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self.L / np.sqrt(2))
            y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * (self.L / np.sqrt(2))
        elif self.DRONE_MODEL == DroneModel.CF2P:
            x_torque = (forces[1] - forces[3]) * self.L
            y_torque = (-forces[0] + forces[2]) * self.L
        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, np.dot(self.J, rpy_rates))
        rpy_rates_deriv = np.dot(self.J_INV, torques)
        no_pybullet_dyn_accs = force_world_frame / self.MASS
        # Update state.
        vel = vel + self.PYB_TIMESTEP * no_pybullet_dyn_accs
        rpy_rates = rpy_rates + self.PYB_TIMESTEP * rpy_rates_deriv
        pos = pos + self.PYB_TIMESTEP * vel
        rpy = rpy + self.PYB_TIMESTEP * rpy_rates
        # Set PyBullet's state.
        p.resetBasePositionAndOrientation(self.DRONE_IDS[nth_drone],
                                          pos,
                                          p.getQuaternionFromEuler(rpy),
                                          physicsClientId=self.PYB_CLIENT)
        # Note: the base's velocity only stored and not used #
        p.resetBaseVelocity(
            self.DRONE_IDS[nth_drone],
            vel,
            rpy_rates,  # ang_vel not computed by DYN
            physicsClientId=self.PYB_CLIENT)
        # Store the roll, pitch, yaw rates for the next step #
        self.rpy_rates[nth_drone, :] = rpy_rates

    def _dynamics_rk4(self, rpm, nth_drone):
        '''Explicit dynamics implementation.

        Based on code written at the Dynamic Systems Lab by James Xu.

        Args:
            rpm (ndarray): (4)-shaped array of ints containing the RPMs values of the 4 motors.
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.
        '''
        # Current state.
        pos = self.pos[nth_drone, :]
        # quat = self.quat[nth_drone, :]
        rpy = self.rpy[nth_drone, :]
        vel = self.vel[nth_drone, :]
        rpy_rates = self.rpy_rates[nth_drone, :]
        # rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        # Compute forces and torques.
        forces = np.array(rpm ** 2) * self.KF
        # Update state with discrete time dynamics.
        state = np.hstack([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2],
                           rpy[0], rpy[1], rpy[2], rpy_rates[0], rpy_rates[1], rpy_rates[2]])
        input = np.array([forces[0], forces[1], forces[2], forces[3]])

        # update state with RK4
        next_state = self.fd_func(x0=state, p=input)['xf']

        pos = np.array([next_state[0], next_state[2], next_state[4]])
        rpy = np.array([next_state[6], next_state[7], next_state[8]])
        vel = np.array([next_state[1], next_state[3], next_state[5]])
        rpy_rates = np.array([next_state[9], next_state[10], next_state[11]])
        rpy_rates = np.squeeze(rpy_rates)

        # Set PyBullet's state.
        p.resetBasePositionAndOrientation(self.DRONE_IDS[nth_drone],
                                          pos,
                                          p.getQuaternionFromEuler(rpy),
                                          physicsClientId=self.PYB_CLIENT)
        # Note: the base's velocity only stored and not used #
        p.resetBaseVelocity(
            self.DRONE_IDS[nth_drone],
            vel,
            rpy_rates,  # ang_vel not computed by DYN
            physicsClientId=self.PYB_CLIENT)
        # Store the roll, pitch, yaw rates for the next step #
        self.rpy_rates[nth_drone, :] = rpy_rates

    def setup_rk4_dynamics_expression(self):
        gamma = self.KM / self.KF
        z = cs.MX.sym('z')
        z_dot = cs.MX.sym('z_dot')
        x = cs.MX.sym('x')
        y = cs.MX.sym('y')
        phi = cs.MX.sym('phi')  # Roll
        theta = cs.MX.sym('theta')  # Pitch
        psi = cs.MX.sym('psi')  # Yaw
        x_dot = cs.MX.sym('x_dot')
        y_dot = cs.MX.sym('y_dot')
        p_body = cs.MX.sym('p')  # Body frame roll rate
        q_body = cs.MX.sym('q')  # body frame pith rate
        r_body = cs.MX.sym('r')  # body frame yaw rate
        # PyBullet Euler angles use the SDFormat for rotation matrices.
        Rob = csRotXYZ(phi, theta, psi)  # rotation matrix transforming a vector in the body frame to the world frame.

        # Define state variables.
        X = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body)

        # Define inputs.
        f1 = cs.MX.sym('f1')
        f2 = cs.MX.sym('f2')
        f3 = cs.MX.sym('f3')
        f4 = cs.MX.sym('f4')
        U = cs.vertcat(f1, f2, f3, f4)

        # From Ch. 2 of Luis, Carlos, and Jérôme Le Ny. 'Design of a trajectory tracking controller for a
        # nanoquadcopter.' arXiv preprint arXiv:1608.05786 (2016).

        # Defining the dynamics function.
        # We are using the velocity of the base wrt to the world frame expressed in the world frame.
        # Note that the reference expresses this in the body frame.
        oVdot_cg_o = Rob @ cs.vertcat(0, 0, f1 + f2 + f3 + f4) / self.MASS - cs.vertcat(0, 0, self.GRAVITY_ACC)
        pos_ddot = oVdot_cg_o
        pos_dot = cs.vertcat(x_dot, y_dot, z_dot)
        Mb = cs.vertcat(self.L / cs.sqrt(2.0) * (f1 + f2 - f3 - f4),
                        self.L / cs.sqrt(2.0) * (-f1 + f2 + f3 - f4),
                        gamma * (-f1 + f2 - f3 + f4))
        rate_dot = self.J_INV @ (
            Mb - (cs.skew(cs.vertcat(p_body, q_body, r_body)) @ self.J @ cs.vertcat(p_body, q_body, r_body)))
        ang_dot = cs.blockcat([[1, cs.sin(phi) * cs.tan(theta), cs.cos(phi) * cs.tan(theta)],
                               [0, cs.cos(phi), -cs.sin(phi)],
                               [0, cs.sin(phi) / cs.cos(theta), cs.cos(phi) / cs.cos(theta)]]) @ cs.vertcat(p_body,
                                                                                                            q_body,
                                                                                                            r_body)
        X_dot = cs.vertcat(pos_dot[0], pos_ddot[0], pos_dot[1], pos_ddot[1], pos_dot[2], pos_ddot[2], ang_dot, rate_dot)
        self.X_dot_fun = cs.Function('X_dot', [X, U], [X_dot])
        self.fd_func = cs.integrator('fd', 'rk', {'x': X,
                                                  'p': U,
                                                  'ode': X_dot}, {'tf': self.PYB_TIMESTEP})

    def _dynamics_2d(self, rpm, nth_drone):
        '''Explicit dynamics implementation.
        Based on the physics model.

        Args:
            rpm (ndarray): (4)-shaped array of ints containing the RPMs values of the 4 motors.
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.
        '''
        # Current state.
        pos = self.pos[nth_drone, :]
        # quat = self.quat[nth_drone, :]
        rpy = self.rpy[nth_drone, :]
        vel = self.vel[nth_drone, :]
        ang_v = self.ang_v[nth_drone, :]

        # Compute forces and torques.
        forces = np.array(rpm ** 2) * self.KF
        # Update state with discrete time dynamics.
        state = np.hstack([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2],
                           rpy[0], rpy[1], rpy[2], ang_v[0], ang_v[1], ang_v[2]])
        action = np.array([forces[0], forces[1], forces[2], forces[3]])

        # update state with RK4
        # next_state = self.fd_func(x0=state, p=input)['xf'].full()[:, 0]
        X_dot = self.X_dot_fun(state, action).full()[:, 0]
        next_state = state + X_dot * self.PYB_TIMESTEP

        # Updated information
        pos = np.array([next_state[0], 0, next_state[4]])
        rpy = np.array([0, next_state[7], 0])
        vel = np.array([next_state[1], 0, next_state[5]])
        ang_v = np.array([0, next_state[10], 0])
        ang_v = np.squeeze(ang_v)

        # # Set PyBullet's state.
        # p.resetBasePositionAndOrientation(self.DRONE_IDS[nth_drone],
        #                                   pos,
        #                                   p.getQuaternionFromEuler(rpy),
        #                                   physicsClientId=self.PYB_CLIENT)
        # # Note: the base's velocity only stored and not used #
        # p.resetBaseVelocity(self.DRONE_IDS[nth_drone],
        #                     vel,
        #                     ang_v,  # ang_vel not computed by DYN
        #                     physicsClientId=self.PYB_CLIENT)
        self.pos[nth_drone, :] = pos.copy()
        self.rpy[nth_drone, :] = rpy.copy()
        self.vel[nth_drone, :] = vel.copy()
        self.ang_v[nth_drone, :] = ang_v.copy()
        # Store the roll, pitch, yaw rates for the next step #
        self.rpy_rates[nth_drone, :] = X_dot[6:9]

    def setup_dynamics_2d_expression(self):
        gamma = self.KM / self.KF

        # Casadi states
        z = cs.MX.sym('z')
        x = cs.MX.sym('x')
        y = cs.MX.sym('y')
        z_dot = cs.MX.sym('z_dot')
        x_dot = cs.MX.sym('x_dot')
        y_dot = cs.MX.sym('y_dot')
        phi = cs.MX.sym('phi')  # Roll
        theta = cs.MX.sym('theta')  # Pitch
        psi = cs.MX.sym('psi')  # Yaw
        p_body = cs.MX.sym('p')  # Body frame roll rate
        q_body = cs.MX.sym('q')  # body frame pitch rate
        r_body = cs.MX.sym('r')  # body frame yaw rate
        X = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body)

        # Define inputs.
        f1 = cs.MX.sym('f1')
        f2 = cs.MX.sym('f2')
        f3 = cs.MX.sym('f3')
        f4 = cs.MX.sym('f4')
        U = cs.vertcat(f1, f2, f3, f4)

        # PyBullet Euler angles use the SDFormat for rotation matrices.
        Rob = csRotXYZ(phi, theta, psi)  # rotation matrix transforming a vector in the body frame to the world frame.

        # From Ch. 2 of Luis, Carlos, and Jérôme Le Ny. 'Design of a trajectory tracking controller for a
        # nanoquadcopter.' arXiv preprint arXiv:1608.05786 (2016).

        # Defining the dynamics function.
        # We are using the velocity of the base wrt to the world frame expressed in the world frame.
        # Note that the reference expresses this in the body frame.
        pos_ddot = Rob @ cs.vertcat(0, 0, f1 + f2 + f3 + f4) / self.MASS - cs.vertcat(0, 0, self.GRAVITY_ACC)
        pos_dot = cs.vertcat(x_dot, y_dot, z_dot)
        Mb = cs.vertcat(self.L / cs.sqrt(2.0) * (f1 + f2 - f3 - f4),
                        self.L / cs.sqrt(2.0) * (-f1 + f2 + f3 - f4),
                        gamma * (-f1 + f2 - f3 + f4))
        rate_dot = self.J_INV @ (
            Mb - (cs.skew(cs.vertcat(p_body, q_body, r_body)) @ self.J @ cs.vertcat(p_body, q_body, r_body)))
        ang_dot = (cs.blockcat([[1, cs.sin(phi) * cs.tan(theta), cs.cos(phi) * cs.tan(theta)],
                                [0, cs.cos(phi), -cs.sin(phi)],
                                [0, cs.sin(phi) / cs.cos(theta), cs.cos(phi) / cs.cos(theta)]]) @
                   cs.vertcat(p_body, q_body, r_body))
        X_dot = cs.vertcat(pos_dot[0], pos_ddot[0], pos_dot[1], pos_ddot[1], pos_dot[2], pos_ddot[2], ang_dot, rate_dot)
        self.X_dot_fun = cs.Function('X_dot', [X, U], [X_dot])

    def _dynamics_si(self, action, nth_drone, disturbance_force=None):
        '''Explicit dynamics implementation from the identified model.
           NOTE: The dynamcis update is independent of the pybullet simulation.

        Args:
            action (ndarray): (2)-shaped array of ints containing the desired collective thrust and pitch.
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.
            disturbance_force (ndarray): (3)-shaped array of floats containing the disturbance force.
                                         with the format [f_x, 0, f_z].
        '''
        # Current state.
        pos = self.pos[nth_drone, :]
        # quat = self.quat[nth_drone, :]
        rpy = self.rpy[nth_drone, :]
        vel = self.vel[nth_drone, :]
        rpy_rates = self.rpy_rates[nth_drone, :]

        # Compute forces and torques.
        # Update state with discrete time dynamics.
        state = np.hstack([pos[0], vel[0], pos[2], vel[2], rpy[1], rpy_rates[1]])

        # update state
        if disturbance_force is not None:
            d = np.array([disturbance_force[0], disturbance_force[2]])
        else:
            d = np.array([0, 0])
        # perform euler integration
        # next_state = state + self.PYB_TIMESTEP * self.X_dot_fun(state, action, d).full()[:, 0]
        # perform RK4 integration
        k1 = self.X_dot_fun(state, action, d).full()[:, 0]
        k2 = self.X_dot_fun(state + 0.5 * self.PYB_TIMESTEP * k1, action, d).full()[:, 0]
        k3 = self.X_dot_fun(state + 0.5 * self.PYB_TIMESTEP * k2, action, d).full()[:, 0]
        k4 = self.X_dot_fun(state + self.PYB_TIMESTEP * k3, action, d).full()[:, 0]
        next_state = state + (self.PYB_TIMESTEP / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Updated information
        pos = np.array([next_state[0], 0, next_state[2]])
        rpy = np.array([0, next_state[4], 0])
        vel = np.array([next_state[1], 0, next_state[3]])
        rpy_rates = np.array([0, next_state[5], 0])

        self.pos[nth_drone, :] = pos.copy()
        self.rpy[nth_drone, :] = rpy.copy()
        self.vel[nth_drone, :] = vel.copy()
        self.rpy_rates[nth_drone, :] = rpy_rates.copy()
        self.ang_v[nth_drone, :] = get_angularvelocity_rpy(self.rpy[nth_drone, :], self.rpy_rates[nth_drone, :])

    def setup_dynamics_si_expression(self, prop_values=None):
        # Casadi states
        z = cs.MX.sym('z')
        x = cs.MX.sym('x')
        z_dot = cs.MX.sym('z_dot')
        x_dot = cs.MX.sym('x_dot')
        theta = cs.MX.sym('theta')  # Pitch
        theta_dot = cs.MX.sym('theta_dot')  # Pitch
        X = cs.vertcat(x, x_dot, z, z_dot, theta, theta_dot)
        g = self.GRAVITY_ACC
        d = cs.MX.sym('d', 2, 1)  # disturbance force

        # Define inputs.
        T = cs.MX.sym('T')  # normlized thrust [N]
        P = cs.MX.sym('P')  # desired pitch angle [rad]
        U = cs.vertcat(T, P)
        if prop_values is None:
            X_dot = cs.vertcat(x_dot,
                               (17.7903 * T + 4.2209) * cs.sin(theta) + d[0] / self.MASS,
                               z_dot,
                               (17.7903 * T + 4.2209) * cs.cos(theta) - g + d[1] / self.MASS,
                               theta_dot,
                               -149.740766 * theta - 16.755920 * theta_dot + 129.041538 * P)
        else:
            X_dot = cs.vertcat(x_dot,
                               (prop_values['beta_1'] * T + prop_values['beta_2']) * cs.sin(theta) + d[0] / self.MASS,
                               z_dot,
                               (prop_values['beta_1'] * T + prop_values['beta_2']) * cs.cos(theta) - g + d[
                                   1] / self.MASS,
                               theta_dot,
                               prop_values['alpha_1'] * theta + prop_values['alpha_2'] * theta_dot + prop_values[
                                   'alpha_3'] * P)
        self.X_dot_fun = cs.Function('X_dot', [X, U, d], [X_dot])

    def _dynamics_si_3d(self, action, nth_drone, disturbance_force=None):
        '''Explicit dynamics implementation from the identified model.
           NOTE: The dynamics update is independent of the pybullet simulation.
        Args:
            action (ndarray): (4)-shaped array of ints containing the desired collective thrust, roll, pitch and yaw.
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.
            disturbance_force (ndarray): (3)-shaped array of floats containing the disturbance force.
                                         with the format [f_x, f_y, f_z].
        '''
        # Current state.
        pos = self.pos[nth_drone, :]
        # quat = self.quat[nth_drone, :]
        rpy = self.rpy[nth_drone, :]
        vel = self.vel[nth_drone, :]
        rpy_rates = self.rpy_rates[nth_drone, :]

        # Compute forces and torques.
        # Update state with discrete time dynamics.
        state = np.hstack([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2],
                           rpy[0], rpy[1], rpy[2], rpy_rates[0], rpy_rates[1], rpy_rates[2]])

        # update state
        if disturbance_force is not None:
            d = np.array([disturbance_force[0], disturbance_force[1], disturbance_force[2]])
        else:
            d = np.array([0, 0, 0])
        # perform euler integration
        # next_state = state + self.PYB_TIMESTEP * self.X_dot_fun(state, action, d).full()[:, 0]
        # perform RK4 integration
        k1 = self.X_dot_fun(state, action, d).full()[:, 0]
        k2 = self.X_dot_fun(state + 0.5 * self.PYB_TIMESTEP * k1, action, d).full()[:, 0]
        k3 = self.X_dot_fun(state + 0.5 * self.PYB_TIMESTEP * k2, action, d).full()[:, 0]
        k4 = self.X_dot_fun(state + self.PYB_TIMESTEP * k3, action, d).full()[:, 0]
        next_state = state + (self.PYB_TIMESTEP / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Updated information
        pos = np.array([next_state[0], next_state[2], next_state[4]])
        vel = np.array([next_state[1], next_state[3], next_state[5]])
        rpy = np.array([next_state[6], next_state[7], next_state[8]])
        rpy_rates = np.array([next_state[9], next_state[10], next_state[11]])

        self.pos[nth_drone, :] = pos.copy()
        self.rpy[nth_drone, :] = rpy.copy()
        self.vel[nth_drone, :] = vel.copy()
        # self.ang_v[nth_drone, :] = ang_v.copy()
        self.rpy_rates[nth_drone, :] = rpy_rates.copy()
        self.ang_v[nth_drone, :] = get_angularvelocity_rpy(self.rpy[nth_drone, :], self.rpy_rates[nth_drone, :])

    def setup_dynamics_si_3d_expression(self, prop_values=None):
        # Casadi states
        x = cs.MX.sym('x')
        y = cs.MX.sym('y')
        z = cs.MX.sym('z')
        x_dot = cs.MX.sym('x_dot')
        y_dot = cs.MX.sym('y_dot')
        z_dot = cs.MX.sym('z_dot')
        phi = cs.MX.sym('phi')  # Roll
        theta = cs.MX.sym('theta')  # Pitch
        psi = cs.MX.sym('psi')  # Yaw
        phi_dot = cs.MX.sym('phi_dot')  # Roll
        theta_dot = cs.MX.sym('theta_dot')  # Pitch
        psi_dot = cs.MX.sym('psi_dot')  # Yaw

        X = cs.vertcat(x, x_dot, y, y_dot, z, z_dot,
                       phi, theta, psi, phi_dot, theta_dot, psi_dot)
        g = self.GRAVITY_ACC
        d = cs.MX.sym('d', 3, 1)  # disturbance force

        # Define inputs.
        T = cs.MX.sym('T')  # normalized thrust [N]
        R = cs.MX.sym('R')  # desired roll angle [rad]
        P = cs.MX.sym('P')  # desired pitch angle [rad]
        Y = cs.MX.sym('Y')  # desired yaw angle [rad]
        U = cs.vertcat(T, R, P, Y)

        # X_dot = cs.vertcat(x_dot,
        #                    (20.763637147006943 * T + 3.1610208881218727) * (
        #                                cs.cos(phi) * cs.sin(theta) * cs.cos(psi) + cs.sin(phi) * cs.sin(psi)) + d[
        #                        0] / self.MASS,
        #                    y_dot,
        #                    ((20.763637147006943 * T + 3.1610208881218727) * (
        #                                cs.cos(phi) * cs.sin(theta) * cs.sin(psi) - cs.sin(phi) * cs.cos(psi)) + d[
        #                        1] / self.MASS)*0,
        #                    z_dot,
        #                    (20.763637147006943 * T + 3.1610208881218727) * cs.cos(phi) * cs.cos(theta) - g + d[
        #                        2] / self.MASS,
        #                    phi_dot,
        #                    theta_dot,
        #                    psi_dot,
        #                    (- 25.651473451232217 * phi - 2.5580262532002482 * phi_dot + 17.524089241776338 * R)*0,
        #                    - 61.62863740616216 * theta - 7.205874472066235 * theta_dot + 51.90335491067372 * P,
        #                    (- 12.544174350349687 * psi - 0.012945379372787613 * psi_dot + 43.839961280232046 * Y)*0)
        # TODO: double-check parameters
        # Haocheng's model
        # params_acc = [20.907574256269616, 3.653687545690674]
        # params_roll_rate = [-130.3, -16.33, 119.3]
        # params_pitch_rate = [-99.94, -13.3, 84.73]
        # params_yaw_rate = [0., 0., 0.]
        # Marcel's model
        if prop_values is None:
            params_acc = [32.221212, 0.]
            params_roll_rate = [-286.2, -23.03, 225.6]
            params_pitch_rate = [-286.2, -23.03, 225.6]
            params_yaw_rate = [-192.9, -22.22, 323.5]

            X_dot = cs.vertcat(x_dot,
                               (params_acc[0] * T + params_acc[1]) * (
                                   cs.cos(phi) * cs.sin(theta) * cs.cos(psi) + cs.sin(phi) * cs.sin(psi)) + d[
                                   0] / self.MASS,
                               y_dot,
                               (params_acc[0] * T + params_acc[1]) * (
                                   cs.cos(phi) * cs.sin(theta) * cs.sin(psi) - cs.sin(phi) * cs.cos(psi)),
                               z_dot,
                               (params_acc[0] * T + params_acc[1]) * cs.cos(phi) * cs.cos(theta) - g + d[1] / self.MASS,
                               phi_dot,
                               theta_dot,
                               psi_dot,
                               params_roll_rate[0] * phi + params_roll_rate[1] * phi_dot + params_roll_rate[2] * R,
                               params_pitch_rate[0] * theta + params_pitch_rate[1] * theta_dot + params_pitch_rate[2] * P,
                               params_yaw_rate[0] * psi + params_yaw_rate[1] * psi_dot + params_yaw_rate[2] * Y)
        else:
            X_dot = cs.vertcat(x_dot,
                               (prop_values['beta_1'] * T + prop_values['beta_2']) * (
                                   cs.cos(phi) * cs.sin(theta) * cs.cos(psi) + cs.sin(phi) * cs.sin(psi)) + d[
                                   0] / self.MASS,
                               y_dot,
                               (prop_values['beta_1'] * T + prop_values['beta_2']) * (
                                   cs.cos(phi) * cs.sin(theta) * cs.sin(psi) - cs.sin(phi) * cs.cos(psi)),
                               z_dot,
                               (prop_values['beta_1'] * T + prop_values['beta_2']) * cs.cos(phi) * cs.cos(theta) - g + d[1] / self.MASS,
                               phi_dot,
                               theta_dot,
                               psi_dot,
                               prop_values['alpha_1'] * phi + prop_values['alpha_2'] * phi_dot + prop_values['alpha_3'] * R,
                               prop_values['alpha_4'] * theta + prop_values['alpha_5'] * theta_dot + prop_values['alpha_6'] * P,
                               prop_values['alpha_7'] * psi + prop_values['alpha_8'] * psi_dot + prop_values['alpha_9'] * Y)

        self.X_dot_fun = cs.Function('X_dot', [X, U, d], [X_dot])

    def _dynamics_si_3d_10(self, action, nth_drone, disturbance_force=None):
        '''Explicit dynamics implementation from the identified model.
           NOTE: The dynamics update is independent of the pybullet simulation.
        Args:
            action (ndarray): (4)-shaped array of ints containing the desired collective thrust, roll, pitch.
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.
            disturbance_force (ndarray): (3)-shaped array of floats containing the disturbance force.
                                         with the format [f_x, f_y, f_z].
        '''
        # Current state.
        pos = self.pos[nth_drone, :]
        # quat = self.quat[nth_drone, :]
        rpy = self.rpy[nth_drone, :]
        vel = self.vel[nth_drone, :]
        ang_v = self.ang_v[nth_drone, :]

        # Compute forces and torques.
        # Update state with discrete time dynamics.
        state = np.hstack([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2],
                           rpy[0], rpy[1], ang_v[0], ang_v[1]])

        # update state
        if disturbance_force is not None:
            d = np.array([disturbance_force[0], disturbance_force[1], disturbance_force[2]])
        else:
            d = np.array([0, 0, 0])
        # perform euler integration
        # next_state = state + self.PYB_TIMESTEP * self.X_dot_fun(state, action, d).full()[:, 0]
        # perform RK4 integration
        k1 = self.X_dot_fun(state, action, d).full()[:, 0]
        k2 = self.X_dot_fun(state + 0.5 * self.PYB_TIMESTEP * k1, action, d).full()[:, 0]
        k3 = self.X_dot_fun(state + 0.5 * self.PYB_TIMESTEP * k2, action, d).full()[:, 0]
        k4 = self.X_dot_fun(state + self.PYB_TIMESTEP * k3, action, d).full()[:, 0]
        next_state = state + (self.PYB_TIMESTEP / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Updated information
        pos = np.array([next_state[0], next_state[2], next_state[4]])
        vel = np.array([next_state[1], next_state[3], next_state[5]])
        rpy = np.array([next_state[6], next_state[7], 0])
        ang_v = np.array([next_state[8], next_state[9], 0])

        self.pos[nth_drone, :] = pos.copy()
        self.rpy[nth_drone, :] = rpy.copy()
        self.vel[nth_drone, :] = vel.copy()
        self.ang_v[nth_drone, :] = ang_v.copy()

    def setup_dynamics_si_3d_10_expression(self):
        # Casadi states
        x = cs.MX.sym('x')
        y = cs.MX.sym('y')
        z = cs.MX.sym('z')
        x_dot = cs.MX.sym('x_dot')
        y_dot = cs.MX.sym('y_dot')
        z_dot = cs.MX.sym('z_dot')
        phi = cs.MX.sym('phi')  # Roll
        theta = cs.MX.sym('theta')  # Pitch
        phi_dot = cs.MX.sym('phi_dot')  # Roll
        theta_dot = cs.MX.sym('theta_dot')  # Pitch

        X = cs.vertcat(x, x_dot, y, y_dot, z, z_dot,
                       phi, theta, phi_dot, theta_dot)
        g = self.GRAVITY_ACC
        d = cs.MX.sym('d', 3, 1)  # disturbance force

        # Define inputs.
        T = cs.MX.sym('T')  # normalized thrust [N]
        R = cs.MX.sym('R')  # desired roll angle [rad]
        P = cs.MX.sym('P')  # desired pitch angle [rad]
        U = cs.vertcat(T, R, P)

        # X_dot = cs.vertcat(x_dot,
        #                    (20.763637147006943 * T + 3.1610208881218727) * (
        #                                cs.cos(phi) * cs.sin(theta) * cs.cos(psi) + cs.sin(phi) * cs.sin(psi)) + d[
        #                        0] / self.MASS,
        #                    y_dot,
        #                    ((20.763637147006943 * T + 3.1610208881218727) * (
        #                                cs.cos(phi) * cs.sin(theta) * cs.sin(psi) - cs.sin(phi) * cs.cos(psi)) + d[
        #                        1] / self.MASS)*0,
        #                    z_dot,
        #                    (20.763637147006943 * T + 3.1610208881218727) * cs.cos(phi) * cs.cos(theta) - g + d[
        #                        2] / self.MASS,
        #                    phi_dot,
        #                    theta_dot,
        #                    psi_dot,
        #                    (- 25.651473451232217 * phi - 2.5580262532002482 * phi_dot + 17.524089241776338 * R)*0,
        #                    - 61.62863740616216 * theta - 7.205874472066235 * theta_dot + 51.90335491067372 * P,
        #                    (- 12.544174350349687 * psi - 0.012945379372787613 * psi_dot + 43.839961280232046 * Y)*0)
        # TODO: double-check parameters
        params_acc = [20.907574256269616, 3.653687545690674]
        params_roll_rate = [-130.3, -16.33, 119.3]
        params_pitch_rate = [-99.94, -13.3, 84.73]
        psi = 0

        X_dot = cs.vertcat(x_dot,
                           (params_acc[0] * T + params_acc[1]) * (
                               cs.cos(phi) * cs.sin(theta) * cs.cos(psi) + cs.sin(phi) * cs.sin(psi)) + d[
                               0] / self.MASS,
                           y_dot,
                           (params_acc[0] * T + params_acc[1]) * (
                               cs.cos(phi) * cs.sin(theta) * cs.sin(psi) - cs.sin(phi) * cs.cos(psi)),
                           z_dot,
                           (params_acc[0] * T + params_acc[1]) * cs.cos(phi) * cs.cos(theta) - g + d[1] / self.MASS,
                           phi_dot,
                           theta_dot,
                           params_roll_rate[0] * phi + params_roll_rate[1] * phi_dot + params_roll_rate[2] * R,
                           params_pitch_rate[0] * theta + params_pitch_rate[1] * theta_dot + params_pitch_rate[2] * P,
                           )

        self.X_dot_fun = cs.Function('X_dot', [X, U, d], [X_dot])

    def _dynamics_si_3d_delay(self, action, nth_drone, disturbance_force=None):
        '''Explicit dynamics implementation from the identified model.
           NOTE: The dynamics update is independent of the pybullet simulation.
        Args:
            action (ndarray): (4)-shaped array of ints containing the desired collective thrust, roll, pitch and yaw.
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.
            disturbance_force (ndarray): (3)-shaped array of floats containing the disturbance force.
                                         with the format [f_x, f_y, f_z].
        '''
        # Current state.
        pos = self.pos[nth_drone, :]
        # quat = self.quat[nth_drone, :]
        rpy = self.rpy[nth_drone, :]
        vel = self.vel[nth_drone, :]
        ang_v = self.ang_v[nth_drone, :]
        rpy_rates = self.rpy_rates[nth_drone, :]
        motor_forces = self.motor_forces[nth_drone, :]
        state = np.hstack([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2],
                           rpy[0], rpy[1], rpy[2], ang_v[0], ang_v[1], ang_v[2],
                           motor_forces])

        # update state
        if disturbance_force is not None:
            d = np.array([disturbance_force[0], disturbance_force[1], disturbance_force[2]])
        else:
            d = np.array([0, 0, 0])
        # perform euler integration
        # next_state = state + self.PYB_TIMESTEP * self.X_dot_fun(state, action, d).full()[:, 0]
        # perform RK4 integration
        k1 = self.X_dot_fun(state, action, d).full()[:, 0]
        k2 = self.X_dot_fun(state + 0.5 * self.PYB_TIMESTEP * k1, action, d).full()[:, 0]
        k3 = self.X_dot_fun(state + 0.5 * self.PYB_TIMESTEP * k2, action, d).full()[:, 0]
        k4 = self.X_dot_fun(state + self.PYB_TIMESTEP * k3, action, d).full()[:, 0]
        delta_state = (self.PYB_TIMESTEP / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        next_state = state + delta_state

        # Updated information
        pos = np.array([next_state[0], next_state[2], next_state[4]])
        vel = np.array([next_state[1], next_state[3], next_state[5]])
        rpy = np.array([next_state[6], next_state[7], next_state[8]])
        rpy_rates = np.array([next_state[9], next_state[10], next_state[11]])

        # # normalize motor forces
        # normal_motor_forces = 2 * (motor_forces - f_min) / (f_max - f_min) - 1
        # # apply delta to normalized motor forces
        # next_normal_motor_forces = normal_motor_forces + delta_state[12]
        # # denominate motor forces to raw force space
        # next_motor_forces = (next_normal_motor_forces + 1) * (f_max - f_min) / 2 + f_min
        # motor_forces = next_motor_forces.copy()

        motor_forces = np.array([next_state[12]])
        motor_forces = np.clip(motor_forces, 0.08, 0.45)
        # print(f"Motor forces after dynamics: {motor_forces}")

        self.pos[nth_drone, :] = pos.copy()
        self.rpy[nth_drone, :] = rpy.copy()
        self.vel[nth_drone, :] = vel.copy()
        self.rpy_rates[nth_drone, :] = rpy_rates.copy()
        self.motor_forces[nth_drone, :] = motor_forces.copy()
        self.ang_v[nth_drone, :] = get_angularvelocity_rpy(self.rpy[nth_drone, :], self.rpy_rates[nth_drone, :])

    def setup_dynamics_si_3d_delay_expression(self, prop_values=None):

        overridden_mass = self.MASS
        if prop_values is not None:
            overridden_mass = prop_values.get('M', self.MASS)

        # Casadi states
        x = cs.MX.sym('x')
        y = cs.MX.sym('y')
        z = cs.MX.sym('z')
        x_dot = cs.MX.sym('x_dot')
        y_dot = cs.MX.sym('y_dot')
        z_dot = cs.MX.sym('z_dot')
        phi = cs.MX.sym('phi')  # Roll
        theta = cs.MX.sym('theta')  # Pitch
        psi = cs.MX.sym('psi')  # Yaw
        phi_dot = cs.MX.sym('phi_dot')  # Roll
        theta_dot = cs.MX.sym('theta_dot')  # Pitch
        psi_dot = cs.MX.sym('psi_dot')  # Yaw
        forces_motor = cs.MX.sym('force_motor', 1)

        X = cs.vertcat(x, x_dot, y, y_dot, z, z_dot,
                       phi, theta, psi, phi_dot, theta_dot, psi_dot, forces_motor)

        g = self.GRAVITY_ACC
        d = cs.MX.sym('d', 3, 1)  # disturbance force
        # Define inputs.
        T_c = cs.MX.sym('T')  # normalized thrust [N]
        R_c = cs.MX.sym('R')  # desired roll angle [rad]
        P_c = cs.MX.sym('P')  # desired pitch angle [rad]
        Y_c = cs.MX.sym('Y')  # desired yaw angle [rad]
        U = cs.vertcat(T_c, R_c, P_c, Y_c)

        # Transformation parameters from sys_id with mode 3
        cmd_min = self.transform_params['cmd_min']
        cmd_max = self.transform_params['cmd_max']
        f_min = self.transform_params['f_min']
        f_max = self.transform_params['f_max']
        # f_hover = self.transform_params['f_hover']
        # cmd_hover = self.transform_params['cmd_hover']

        # Transform input command T from raw to normalized space (mode 3)
        # T is expected to be in raw force units, normalize to [-1, 1]
        dT_c = 2 * (T_c - cmd_min) / (cmd_max - cmd_min) - 1

        # normalized forces_motor
        df = 2 * (forces_motor - f_min) / (f_max - f_min) - 1

        # Delay dynamics parameters (from MATLAB sys_id results)
        # Based on estimated parameters: [bias, scale, tau]
        # bias = -0.04 # Update this with actual estimated bias from MATLAB
        # scale = 0.776  # Update this with actual estimated scale from MATLAB
        # tau = 0.092  # Update this with actual estimated tau from MATLAB
        # params_acc = [-0.2039, 0.8, 0.076]  # [bias, scale, tau]
        params_acc = [0.0905, 0.8, 0.0814]
        # Delay dynamics in normalized space: f_dot = (scale * cmd - f) / tau
        # force_motor_dot is the derivative in normalized space
        df_dot = (params_acc[1] * (dT_c + params_acc[0]) - df) / params_acc[2]
        # df_dot = (params_acc[1] * (T_c + params_acc[0]) - forces_motor) / params_acc[2]

        # by definition, motor_forces_dot = 1/2 * df_dot

        # Transform normalized forces_motor to raw force for physics calculations
        # self.df_dot_fun = cs.Function("df_dot", [forces_motor, T], [df_dot])

        # print(f"Using mass: {overridden_mass}")
        # params_acc = [0.5210, 0.1704, 0.0923]
        # params_roll_rate = [-286.2, -23.03, 225.6]
        # params_pitch_rate = [-286.2, -23.03, 225.6]
        # params_yaw_rate = [-192.9, -22.22, 323.5]
        # params_roll_rate = [-1.18e05, -1.0104e-4, 1667]
        # params_pitch_rate = [-255, -17.52, 3.262]
        # params_roll_rate = [-621.8, -55.84, 80.537]
        # params_pitch_rate = [-621.8, -55.84, 80.537]
        # update rpy parameters
        params_roll_rate = [-238.1, -21.35, 179.65]
        params_pitch_rate = [-238.1, -21.35, 179.65]
        params_yaw_rate = [-170.4, -22.22, 280]

        X_dot = cs.vertcat(x_dot,
                           1 / overridden_mass * forces_motor * (cs.cos(phi) * cs.sin(theta) * cs.cos(psi) + cs.sin(phi) * cs.sin(psi)) + d[0] / self.MASS,
                           y_dot,
                           1 / overridden_mass * forces_motor * (cs.cos(phi) * cs.sin(theta) * cs.sin(psi) - cs.sin(phi) * cs.cos(psi)) + d[1] / self.MASS,
                           z_dot,
                           1 / overridden_mass * forces_motor * cs.cos(phi) * cs.cos(theta) - g + d[2] / self.MASS,
                           phi_dot,
                           theta_dot,
                           psi_dot,
                           params_roll_rate[0] * phi + params_roll_rate[1] * phi_dot + params_roll_rate[2] * R_c,
                           params_pitch_rate[0] * theta + params_pitch_rate[1] * theta_dot + params_pitch_rate[2] * P_c,
                           params_yaw_rate[0] * psi + params_yaw_rate[1] * psi_dot + params_yaw_rate[2] * Y_c,
                           (f_max - f_min) / 2 * df_dot)
        self.X_dot_fun = cs.Function('X_dot', [X, U, d], [X_dot])

    def _show_drone_local_axes(self, nth_drone):
        '''Draws the local frame of the n-th drone in PyBullet's GUI.

        Args:
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.
        '''
        if self.GUI:
            AXIS_LENGTH = 2 * self.L
            self.X_AX[nth_drone] = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[AXIS_LENGTH, 0, 0],
                lineColorRGB=[1, 0, 0],
                parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                parentLinkIndex=-1,
                replaceItemUniqueId=int(self.X_AX[nth_drone]),
                physicsClientId=self.PYB_CLIENT)
            self.Y_AX[nth_drone] = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, AXIS_LENGTH, 0],
                lineColorRGB=[0, 1, 0],
                parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                parentLinkIndex=-1,
                replaceItemUniqueId=int(self.Y_AX[nth_drone]),
                physicsClientId=self.PYB_CLIENT)
            self.Z_AX[nth_drone] = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, 0, AXIS_LENGTH],
                lineColorRGB=[0, 0, 1],
                parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                parentLinkIndex=-1,
                replaceItemUniqueId=int(self.Z_AX[nth_drone]),
                physicsClientId=self.PYB_CLIENT)

    def _parse_urdf_parameters(self, file_name):
        '''Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.
        '''
        URDF_TREE = etxml.parse(file_name).getroot()
        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [
            float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')
        ]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        PWM2RPM_SCALE = float(URDF_TREE[0].attrib['pwm2rpm_scale'])
        PWM2RPM_CONST = float(URDF_TREE[0].attrib['pwm2rpm_const'])
        MIN_PWM = float(URDF_TREE[0].attrib['pwm_min'])
        MAX_PWM = float(URDF_TREE[0].attrib['pwm_max'])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
            GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3, \
            PWM2RPM_SCALE, PWM2RPM_CONST, MIN_PWM, MAX_PWM
