import retro

GAME_NAME=""

def main():
    env = retro.make(game=GAME_NAME)
    obs = env.reset()
    while True:
        obs, reward, done, info = env.step(env.action_space.sample())
        env.render()
        print(f"Observation: {obs}\nReward: {reward}\nInfo: {info}")
        if done:
            obs = env.reset()
    env.close()

if __name__=="__main__":
    main()