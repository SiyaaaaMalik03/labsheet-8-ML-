
import os
import random
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

random.seed(42)
np.random.seed(42)

# 0 = free path, 1 = wall
maze = np.array([
    [0, 0, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1],
    [0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0]
])

ROWS, COLS = maze.shape
START = (0, 0)
GOAL = (4, 5)

# Up, Down, Left, Right
ACTIONS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]

NUM_STATES = ROWS * COLS
NUM_ACTIONS = len(ACTIONS)


def state_id(position):
    return position[0] * COLS + position[1]


def is_valid(position):
    row, col = position

    return (
        0 <= row < ROWS
        and 0 <= col < COLS
        and maze[row, col] == 0
    )


def take_action(position, action):
    dr, dc = ACTIONS[action]

    next_position = (
        position[0] + dr,
        position[1] + dc
    )

    # Penalty for wall or boundary
    if not is_valid(next_position):
        return position, -5, False

    # Reward for reaching goal
    if next_position == GOAL:
        return next_position, 20, True

    # Small penalty for each ordinary move
    return next_position, -1, False


def train_agent(episodes=3000):
    q_table = np.zeros((NUM_STATES, NUM_ACTIONS))

    alpha = 0.1
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    rewards = []
    successes = []

    for episode in range(episodes):
        position = START
        total_reward = 0
        done = False

        for _ in range(NUM_STATES * 4):
            current_state = state_id(position)

            # Exploration or best-known action
            if random.random() < epsilon:
                action = random.randrange(NUM_ACTIONS)
            else:
                action = int(
                    np.argmax(q_table[current_state])
                )

            next_position, reward, done = take_action(
                position, action
            )

            next_state = state_id(next_position)

            # Q-learning update
            target = reward

            if not done:
                target += gamma * np.max(
                    q_table[next_state]
                )

            q_table[current_state, action] += alpha * (
                target - q_table[current_state, action]
            )

            position = next_position
            total_reward += reward

            if done:
                break

        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay
        )

        rewards.append(total_reward)
        successes.append(int(done))

    return q_table, rewards, successes


def find_path(q_table):
    position = START
    path = [position]
    visited = {position}

    for _ in range(NUM_STATES * 3):
        if position == GOAL:
            break

        current_state = state_id(position)
        action_order = np.argsort(
            q_table[current_state]
        )[::-1]

        next_position = None

        for action in action_order:
            dr, dc = ACTIONS[int(action)]

            candidate = (
                position[0] + dr,
                position[1] + dc
            )

            if is_valid(candidate) and candidate not in visited:
                next_position = candidate
                break

        if next_position is None:
            break

        position = next_position
        path.append(position)
        visited.add(position)

    return path


def main():
    print("Training Q-learning agent...")

    q_table, rewards, successes = train_agent()

    # Save Q-table
    np.save("outputs/q_table.npy", q_table)

    # Plot training rewards
    window = 100
    moving_average = np.convolve(
        rewards,
        np.ones(window) / window,
        mode="valid"
    )

    plt.figure(figsize=(9, 5))
    plt.plot(rewards, alpha=0.3, label="Episode Reward")
    plt.plot(
        np.arange(window - 1, len(rewards)),
        moving_average,
        label="100-Episode Average"
    )
    plt.title("Q-Learning Training Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/learning_curve.png")
    plt.show()

    # Find learned route
    path = find_path(q_table)
    reached_goal = bool(path and path[-1] == GOAL)

    # Display maze and route
    plt.figure(figsize=(7, 5))
    plt.imshow(maze, cmap="Greys", vmin=0, vmax=1)

    if len(path) > 1:
        plt.plot(
            [p[1] for p in path],
            [p[0] for p in path],
            marker="o",
            label="Learned Path"
        )

    plt.scatter(
        START[1], START[0],
        marker="s", s=150, label="Start"
    )
    plt.scatter(
        GOAL[1], GOAL[0],
        marker="*", s=250, label="Goal"
    )

    plt.xticks(range(COLS))
    plt.yticks(range(ROWS))
    plt.title("Q-Learning Maze Solution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/maze_solution.png")
    plt.show()

    # Evaluate training
    success_rate = np.mean(successes[-100:]) * 100

    with open("outputs/evaluation.txt", "w") as file:
        file.write(f"Episodes: {len(rewards)}\n")
        file.write(
            f"Success rate in final 100 episodes: "
            f"{success_rate:.2f}%\n"
        )
        file.write(f"Path reaches goal: {reached_goal}\n")
        file.write(f"Learned path: {path}\n")

    print("Success rate in final 100 episodes:",
          round(success_rate, 2), "%")
    print("Path reaches goal:", reached_goal)
    print("Q-table and plots saved in outputs folder.")


if __name__ == "__main__":
    main()
