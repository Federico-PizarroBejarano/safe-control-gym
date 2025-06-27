import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.constants import GRAVITY, MASS
from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.symbolic import symbolic_from_sim
from jax.scipy.spatial.transform import Rotation as RotLib

from safe_control_gym.controllers.lqr.lqr_utils import compute_lqr_gain, get_cost_weight_matrix


def lqr_control(obs, t, gain, U_EQ):
    circle = jnp.array([jnp.cos(t) - 1, jnp.sin(t), 0.2 * t] + [0] * 9)

    rpy = RotLib.from_quat(obs.quat).as_euler('xyz').reshape(1, 1, 3)
    stacked_obs = jnp.concatenate([obs.pos, rpy, obs.vel, obs.ang_vel], axis=-1).flatten()

    cmd = -gain @ (stacked_obs - circle) + U_EQ
    cmd = jnp.clip(cmd, jnp.array([0.09702, -0.7, -0.7, -0.7]), jnp.array([0.58212, 0.7, 0.7, 0.7]))
    return cmd


def get_LQR_gain(model, q_lqr, r_lqr, U_EQ, discrete_dynamics=True):
    Q = get_cost_weight_matrix(q_lqr, model.nx)
    R = get_cost_weight_matrix(r_lqr, model.nu)

    gain = compute_lqr_gain(model, np.zeros((12,)), U_EQ, Q, R, discrete_dynamics)
    return np.squeeze(gain)


def main(duration=5.0, fps=60):
    sim = Sim(n_drones=1, control=Control.attitude, integrator='rk4', physics='analytical')
    sim.reset()
    dt = 1.0 / sim.control_freq

    model = symbolic_from_sim(sim)
    q_lqr = [1.0] * 3 + [0.1] * 9
    r_lqr = [0.1] * 4
    U_EQ = np.array([GRAVITY * MASS, 0, 0, 0])
    gain = get_LQR_gain(model, q_lqr, r_lqr, U_EQ, discrete_dynamics=True)

    jit_lqr_control = jax.jit(lqr_control)

    for i in range(int(duration * sim.control_freq)):
        obs = sim.data.states
        cmd = jit_lqr_control(obs, i * dt, gain, U_EQ)
        sim.attitude_control(cmd.reshape(1, 1, -1))
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
    sim.close()


if __name__ == '__main__':
    main()
