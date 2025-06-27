import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.control import Control
from crazyflow.control.control import state2attitude
from crazyflow.sim import Sim

jit_state2attitude = jax.jit(state2attitude)


def control(obs, t, i_error, dt):
    pos, vel, quat = obs.pos, obs.vel, obs.quat
    des_pos = jnp.array([jnp.cos(t) - 1, jnp.sin(t), 0.2 * t])
    cmd, i_error = jit_state2attitude(
        pos, vel, quat, des_pos, np.zeros((1, 1, 3)), np.zeros((1, 1, 1)), i_error, dt
    )
    return cmd, i_error


def main(duration=5.0, fps=60):
    sim = Sim(n_drones=1, control=Control.attitude, attitude_freq=25, integrator='rk4', physics='analytical')
    sim.reset()
    dt = 1 / sim.control_freq

    i_error = np.zeros((1, 1, 3))
    for i in range(int(duration * sim.control_freq)):
        obs = sim.data.states
        cmd, i_error = control(obs, i * dt, i_error, dt)
        sim.attitude_control(cmd.reshape(1, 1, -1))
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
    sim.close()


if __name__ == '__main__':
    main()
