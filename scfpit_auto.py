import Arena
from MCTS import MCTS
from scf.ScfGame import ScfGame, display
from scf.ScfPlayers import RandomPlayer, HumanPlayer, AbPlayer
from scf.keras.NNet import NNetWrapper as NNet

import numpy as np
from utils import dotdict
import os
import pandas as pd
import re
from datetime import datetime

df = pd.DataFrame(columns = ['checkpoint', 'saved_time', 'win', 'loss', 'draw'])
for curDir, dirs, files in os.walk("./temp"):
    for f in files:
        # filename = re.findall('checkpoint_.*.h5', f)
        filename = re.findall('checkpoint_.*.pth.tar', f)

        if (len(filename)):    
            fName = filename[0]
            cp_num = filename[0]
            
            cp_num = cp_num.replace("checkpoint_", "")
            # cp_num = cp_num.replace(".h5", "")
            cp_num = cp_num.replace(".pth.tar", "")

            # compA = "checkpoint_" + str(cp_num) + ".h5"
            compA = "checkpoint_" + str(cp_num) + ".pth.tar"
            if (not compA == f):
                continue

            state = os.stat('./temp/{}'.format(fName))
            saved_time = datetime.fromtimestamp(state.st_mtime)
            
            g = ScfGame()

            # all players
            randPlayer = RandomPlayer(g)
            humanPlayer = HumanPlayer(g)
            # αβ探索をするプレーヤ。時間との兼ね合いで探索深さを設定
            #abPlayer = AbPlayer(g, 10)
            #abPlayer = AbPlayer(g, 6)
            abPlayer = AbPlayer(g, 2)

            # nnet players
            nn = NNet(g)
            # 読み込む学習データを指定する
            # nn.load_checkpoint('./temp/', 'best.pth.tar')
            nn.load_checkpoint('./temp/', f)
            args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
            mcts1 = MCTS(g, nn, args1)
            nnPlayer = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

            numGames = 100  # 10, 100, 1000
            arena = Arena.Arena(nnPlayer, abPlayer.play, g, display=display)
            print("Match " + str(numGames) + ": PlayerA=nn PlayerB=" + str(abPlayer))
            result = arena.playGames(numGames, verbose=True)
            print()
            print('(Awin, Bwin, draw)=', result)
            (win, loss, draw) = result

            addRow = pd.DataFrame([cp_num, saved_time, win, loss, draw], index=df.columns).T
            df = df.append(addRow)

# print()
# print(df)
df = df.sort_values('saved_time')
df.to_csv('./temp.csv', index=False)
