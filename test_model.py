import numpy as np
import os
import random
from cnn import *
from Gomoku_v8 import Gomoku
from mcts import MCTS

env = Gomoku()
ai_turn = 0
bw, ww, draw = False, False, False
model = load_model('models/Gomoku_{}x{}-{}.txt'.format(env.size,env.size,env.win))

env.reset()
mcts = MCTS()
po = 800
sp = False
env.render()
while not (bw or ww or draw):
    if env.t == ai_turn or sp:
        for _ in range(po):
            
            mcts.search(env, model)
            
        winrate = max(mcts.Q[str(env.b)])/2 + 0.5
        print('winrate: '+ str(winrate))
        pi = mcts.N[str(env.b)]
#        pi = model.predict(np.array([env.get_state()]))[0]
        action = np.argmax(pi)
        
        bw, ww, draw = env.ai_step(action)
       
        env.render()
    else:
        x = int(input('x: '))
        y = int(input('y: '))
        bw, ww, draw = env.step(x,y)
        env.render()
    if bw:
        print('bw')
    elif ww:
        print('ww')
    elif draw:
        print('draw')
        
    
