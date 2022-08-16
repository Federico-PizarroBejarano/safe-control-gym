'''Base controller. '''

from abc import ABC, abstractmethod

import torch


class BaseController(ABC):
    '''Template for controller/agent, implement the following methods as needed. '''

    def __init__(self,
                 env_func,
                 training=True,
                 checkpoint_path='temp/model_latest.pt',
                 output_dir='temp',
                 use_gpu=False,
                 seed=0,
                 **kwargs
                 ):
        '''Initializes controller agent.

        Args:
            env_func (callable): function to instantiate task/env.
            training (bool): training flag.
            checkpoint_path (str): file to save trained model & experiment state.
            output_dir (str): folder to write outputs.
            use_gpu (bool): False (use cpu) True (use cuda).
            seed (int): random seed.
        '''

        # Base args.
        self.env_func = env_func
        self.training = training
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = 'cpu' if self.use_gpu is False else 'cuda'
        self.seed = seed
        self.safety_filter = None

        # Algorithm specific args.
        for key, value in kwargs.items():
            self.__dict__[key] = value

        self.setup_results_dict()

    @abstractmethod
    def select_action(self, obs, info=None):
        '''Determine the action to take at the current timestep.

        Args:
            obs (ndarray): the observation at this timestep.
            info (list): the info at this timestep.

        Returns:
            action (ndarray): the action chosen by the controller.
        '''
        return

    def extract_step(self, info=None):
        '''Extracts the current step from the info.

        Args:
            info (list): the info list returned from the environment.

        Returns:
            step (int): the current step/iteration of the environment.
        '''

        if info is not None:
            step = info['current_step']
        else:
            step = 0

        return step

    def learn(self,
              env=None,
              **kwargs
              ):
        '''Performs learning (pre-training, training, fine-tuning, etc).

        Args:
            env (gym.Env): the environment to be used for training
        '''
        return

    def reset(self):
        '''Do initializations for training or evaluation. '''
        return

    def reset_before_run(self, obs=None, info=None, env=None):
        '''Reinitialize just the controller before a new run.

        Args:
            obs (ndarray): the initial observation for the new run
            info (list): the first info of the new run
            env (gym.Env): the environment to be used for the new run
        '''
        self.setup_results_dict()

    def close(self):
        '''Shuts down and cleans up lingering resources. '''
        return

    def save(self,
             path
             ):
        '''Saves model params and experiment state to checkpoint path.

        Args:
            path (str): the path where to save the model params/experiment state
        '''
        return

    def load(self,
             path
             ):
        '''Restores model and experiment given checkpoint path.

        Args:
            path (str): the path where the model params/experiment state are saved
        '''
        return

    def setup_results_dict(self):
        '''Setup the results dictionary to store run information. '''
        self.results_dict = {}
