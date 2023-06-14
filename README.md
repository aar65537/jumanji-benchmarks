# Jumanji Benchmarks

This project contains benchmarks for the jumanji 2048 environment.

## Dashboard

To view a dashboard displaying all benchmark results run the following command and navigate to [localhost:8050](http://localhost:8050/).

```
docker run -it --rm -p 8050:8050 ghcr.io/aar65537/jumanji-benchmarks:main
```

## Improvements

<div style="margin-left: auto;
            margin-right: auto;
            width: min(24rem, 100%)">

|      | no vmap | vmap 10<sup>3</sup> | vmap 10<sup>6</sup> |
|:-----|-:|-:|-:|
| cpu  | 64.36%  | 201.80%  | 392.29% |
| cuda | 900.12% | 1923.08% | 706.87% |
</div>

The above figure shows the total performance increase with all changes (measured as percent increase in steps/sec). The no vmap environments were wrapped with `AutoResetWrapper`. The cpu vmap environments were wrapped with `VmapAutoResetWrapper`. The cuda vmap environments were wrapped with `AutoResetWrapper` and then `VmapWrapper`. I found that `VmapAutoResetWrapper` had poor performance on the gpu.

The improvements fall into three main categories: minimizing conditional logic, preferring `jax.vmap` over `jax.lax` control flow, and algorithmic improvements. Minimizing conditional logic is important because when wrapped with `jax.vmap`, all branches of a conditional expression will be evaluated. Using `jax.vmap` instead of `jax.lax` control flow seems to reduce overhead when running on the gpu. Algorithmic improvements include an optimized move algorithm and a can move algorithm that doesn't mutate the board.

### 2e9f0186: Remove call to `move` inside `jax.lax.switch`

<div style="margin-left: auto;
            margin-right: auto;
            width: min(24rem, 100%)">

|      | no vmap | vmap 10<sup>3</sup> | vmap 10<sup>6</sup> |
|:-----|-:|-:|-:|
| cpu  | -3.05% | 74.75% | 47.83% |
| cuda | -0.06% | 51.00% | 45.00% |
</div>

The current environment selects the correct move in a step using the following switch statement.
```
updated_board, additional_reward = jax.lax.switch(
    action,
    [move_up, move_right, move_down, move_left],
    state.board,
)
```
The problem is when vectorized all branches of the switch will be evaluated. So each call to step will perform all actions, not just the action you want. The solution is to only transform the board in the switch and perform the move outside of the switch.
```
updated_board, additional_reward = move(state.board, action)

def move(board, action, final_shift = True):
    board = transform_board(board, action)
    board, additional_reward = move_up(board, final_shift)
    board = transform_board(board, action)
    return board, additional_reward

def transform_board(board, action):
    return jax.lax.switch(
        action,
        [
            lambda: board,
            lambda: jnp.flip(jnp.transpose(board)),
            lambda: jnp.flip(board),
            lambda: jnp.transpose(board),
        ],
    )
```
This implementation avoids actually calling the expensive `move_up` inside the switch statement.

### ca2e4ba5: Remove call to `set` inside `jax.lax.cond`

<div style="margin-left: auto;
            margin-right: auto;
            width: min(24rem, 100%)">

|      | no vmap | vmap 10<sup>3</sup> | vmap 10<sup>6</sup> |
|:-----|-:|-:|-:|
| cpu  | -12.21% | 58.71% | 22.45% |
| cuda | -9.24%  | 13.10% | 30.53% |
</div>

Currently, the environment shifts column elements using the following conditional statement.

```
def shift_nonzero_element(col, j, i):
    col = col.at[j].set(col[i])
    return col, j + 1

col, j = jax.lax.cond(
    col[i] != 0,
    shift_nonzero_element,
    lambda col, j, i: col, j,
    col, j, i
)
```

Again, we see that there is unnecessary logic inside a conditional. However in this case the source of the slow down is a a bit more obtuse. The problem is actually the line `col = col.at[j].set(col[i])`. This line is supposed to be performed in place, but since both branches of the conditional must be computed that isn't possible. Instead, a new copy of `col` will be created. This copy can be avoided by mutating the array outside of the conditional.

```
new_col_j, new_j = jax.lax.cond(
    col[i] != 0,
    lambda col, j, i: (col[i], j + 1),
    lambda col, j, i: (col[j], j),
    col, j, i
)
col = col.at[j].set(new_col_j)
```

A similar problem was also present when merging tiles.

### 8f9a67bd: Change `jax.lax.scan` to `jax.vmap`

<div style="margin-left: auto;
            margin-right: auto;
            width: min(24rem, 100%)">

|      | no vmap | vmap 10<sup>3</sup> | vmap 10<sup>6</sup> |
|:-----|-:|-:|-:|
| cpu  | -11.71% | -57.18% | -40.42% |
| cuda | 337.66% | 219.20% | 25.32%  |
</div>

Currently, the environment moves each column of the board with the following scan expression.

```
(board, additional_reward), _ = jax.lax.scan(
    f=functools.partial(move_up_col, final_shift=final_shift),
    init=(board, 0.0),
    xs=jnp.arange(board.shape[0]),
)
```

However, a scan isn't really necessary because each row can be moved independently. In this case, it is possible to rewrite the function so it can be wrapped with `jax.vmap`. I found that this reduced overhead on the gpu.

```
board, additional_reward = jax.vmap(move_left_row, (0, None))(board, final_shift)
```

### 77fccd6e: Implement new `move` algorithm

<div style="margin-left: auto;
            margin-right: auto;
            width: min(24rem, 100%)">

|      | no vmap | vmap 10<sup>3</sup> | vmap 10<sup>6</sup> |
|:-----|-:|-:|-:|
| cpu  | 20.96% | -8.01% | 33.46% |
| cuda | 33.44% | 63.62% | 28.01% |
</div>

I also implemented a new optimized move algorithm that only uses a single while loop.

### f5c34b3c: Implement `can_move` algorithm

<div style="margin-left: auto;
            margin-right: auto;
            width: min(24rem, 100%)">

|      | no vmap | vmap 10<sup>3</sup> | vmap 10<sup>6</sup> |
|:-----|-:|-:|-:|
| cpu  | 80.83% | 176.21% | 242.01% |
| cuda | 88.79% | 126.83% | 165.74% |
</div>

Currently, the environment checks if an action is valid by performing the action and seeing if any tiles changed. 

```
jnp.any(move_up(board, final_shift=False)[0] != board)
```

Even with the optimization of not performing the final shift, this is fairly expensive. I implemented a can move algorithm that can validate an action without mutating the board.
