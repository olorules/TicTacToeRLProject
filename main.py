from DQNPlayer import DQNPlayer

if __name__ == '__main__':
    # DQNPlayer will output tensorboard data to ./tfb_data/*
    # 1. 'pip install tensorboard'
    # 2. in terminal go to TicTacToeRLProject folder
    # 3. 'tensorboard --logdir tfb_data' will start http server with plots
    # 4. open 127.0.0.1:6006 (not 0.0.0.0:6006) in internet browser

    dqn = DQNPlayer(6000)
    dqn.learn()
    #%%
    from QPlayer import QPlayer
    pl = QPlayer(6000)
    pl.learn()
    #%%
    from Game import Game
    #print(pl.rewards_all_episodes)

    q1 = 0
    q2 = 0
    t = 0
    for elem in range(1000):
        gm = Game()
        res = gm.play_game_for_comparison([dqn, pl])
        if(res == 1):
            q1+=1
        elif(res == 2):
            q2+=1
        else:
            t+=1
        print((q1,q2,t))
    print('#########################')
    q1 = 0
    q2 = 0
    t = 0
    for elem in range(1000):
        gm = Game()
        res = gm.play_game_for_comparison([pl, dqn])
        if(res == 1):
            q1+=1
        elif(res == 2):
            q2+=1
        else:
            t+=1
        print((q1,q2,t))