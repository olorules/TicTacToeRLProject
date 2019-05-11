from DQNPlayer import DQNPlayer


if __name__ == '__main__':
    # DQNPlayer will output tensorboard data to ./tfb_data/*
    # 1. 'pip install tensorboard'
    # 2. in terminal go to TicTacToeRLProject folder
    # 3. 'tensorboard --logdir tfb_data' will start http server with plots
    # 4. open 127.0.0.1:6006 (not 0.0.0.0:6006) in internet browser

    # pl = QPlayer()
    pl = DQNPlayer()
    pl.learn()
    print(pl.rewards_all_episodes)
