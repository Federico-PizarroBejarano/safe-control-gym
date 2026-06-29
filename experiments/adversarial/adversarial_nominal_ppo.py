'''PPO variant that replaces the RL reward with adversarial nominal-command shaping.'''

import numpy as np

from experiments.adversarial.adversarial_reward import (AdversarialNominalConfig,
                                                        compute_adversarial_rewards_training_batch)
from safe_control_gym.controllers.ppo.ppo import PPO


class AdversarialNominalPPO(PPO):
    '''Train a policy to adversarially challenge NL-MPSC via shaped rewards (no plant disturbances).'''

    def __init__(self, env_func, training=True, checkpoint_path='model_latest.pt',
                 output_dir='temp', use_gpu=True, seed=0, **kwargs):
        adversarial_reward = kwargs.pop('adversarial_reward', None)
        super().__init__(env_func, training, checkpoint_path, output_dir, use_gpu, seed, **kwargs)
        self.adversarial_cfg = AdversarialNominalConfig.from_mapping(adversarial_reward)

    def _policy_env_ref(self):
        ewrap = self.env
        if hasattr(ewrap, 'envs'):
            return ewrap.envs[0]
        if hasattr(ewrap, 'env'):
            return ewrap.env
        return ewrap

    def process_step_reward(
            self, obs, next_obs, action, info, rew,
            physical_action=None, certified_action=None, success=False):
        ev = self._policy_env_ref()
        adversarial_rew = compute_adversarial_rewards_training_batch(
            obs, next_obs, action, info,
            physical_action, certified_action, success, ev, self.adversarial_cfg)
        return np.asarray(adversarial_rew, dtype=np.float32).reshape(np.asarray(rew).shape)
