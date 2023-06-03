from typing import Any, Dict, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from jumanji.env import Environment
from jumanji.environments.logic.game_2048.env import Game2048
from jumanji.environments.logic.game_2048.types import State
from jumanji.wrappers import AutoResetWrapper, VmapAutoResetWrapper, VmapWrapper


@register_pytree_node_class
class RolloutRunner:
    def __init__(
        self,
        key: chex.PRNGKey,
        total_steps: Union[int, chex.Array],
        batch_size: int,
        double_wrap: bool,
    ) -> None:
        self.key = key
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.double_wrap = double_wrap

        # Verify environment doesn't throw error
        self.env

        # Verify n_steps greater than 0
        if isinstance(total_steps, int) and self.n_steps <= 0:
            raise NotImplementedError("Total steps must be larger than batch size.")

    @property
    def env(self) -> Environment[State]:
        env: Environment[State] = Game2048()

        if self.batch_size == 0 and self.double_wrap:
            raise NotImplementedError("Cannot double wrap non-vectorized environment.")
        elif self.batch_size == 0:
            env = AutoResetWrapper(env)
        elif self.double_wrap:
            env = VmapWrapper(AutoResetWrapper(env))
        else:
            env = VmapAutoResetWrapper(env)

        return env

    @property
    def n_steps(self) -> Union[int, chex.Array]:
        return self.total_steps // (1 if self.batch_size == 0 else self.batch_size)

    def rollout(self) -> State:
        init_key, rollout_key = jax.random.split(self.key)

        if self.batch_size > 0:
            init_key = jax.random.split(init_key, self.batch_size)

        init_state, _ = self.env.reset(init_key)
        init_val = (rollout_key, init_state)
        _, final_state = jax.lax.fori_loop(0, self.n_steps, self._step, init_val)

        return final_state  # type: ignore

    def _step(
        self, i: int, val: Tuple[chex.PRNGKey, State]
    ) -> Tuple[chex.PRNGKey, State]:
        del i

        key, state = val
        next_key, subkey = jax.random.split(key)
        actions = jnp.arange(4, dtype=int)
        choice = jax.random.choice

        if self.batch_size > 0:
            subkey = jax.random.split(subkey, self.batch_size)
            actions = jnp.tile(actions, (self.batch_size, 1))
            choice = jax.vmap(choice)

        action = choice(subkey, actions, p=state.action_mask)
        next_state, _ = self.env.step(state, action)

        return next_key, next_state

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[chex.PRNGKey, Union[int, chex.Array]], Dict[str, Any]]:
        children = (self.key, self.total_steps)
        aux_data = {"batch_size": self.batch_size, "double_wrap": self.double_wrap}

        return (children, aux_data)

    @classmethod
    def tree_unflatten(
        cls, aux_data: Dict["str", Any], children: Tuple[chex.PRNGKey, chex.Array]
    ) -> "RolloutRunner":
        return cls(*children, **aux_data)
