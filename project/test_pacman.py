from simple_pacman_env import SimplePacmanEnv
import time

def main():
    env = SimplePacmanEnv(grid_size=(5, 5), num_ghosts=1)
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = env.action_space.sample()  # random action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        time.sleep(0.5)

    print("Game finished! Total reward:", total_reward)

if __name__ == "__main__":
    main()
