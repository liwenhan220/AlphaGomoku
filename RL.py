from cnn import *
from mcts import MCTS
import numpy as np
import random
import os
from Gomoku_v8 import Gomoku
from collections import deque

if not os.path.isdir('models'):
    os.mkdir('models')

numIters = 150
numEps = 25
numMCTSSims = 100
#threshold = 0.55 #for pit
minibatch_size = 256
EPOCHS = 10
don = 10

if not os.path.isdir('selfplay'):
    os.mkdir('selfplay')
    set = None
else:
    set = np.load('selfplay/selfplay_set.npy', allow_pickle=True)


def calPI(ls, Temperature):
    if Temperature == 1:
        t = sum(np.array(ls)**(1/Temperature))+1
        return np.array(ls)**(1/Temperature)/t
    else:
        arr = [0 for _ in range(len(ls))]
        a = np.argmax(ls)
        arr[a] = 1
        return arr

def PolicyIterSP(game):
    global set
#    nnet = load_model('models/Gomoku_v1.txt')
    nnet = create_model((game.size,game.size,3),game.size**2)
    trainset = deque(maxlen=10000)
    if set is not None:
        trainset += list(set)
    for i in range(numIters):        
        for e in range(numEps):
            print('Iter: {}'.format(i))
            print('Epi: {}'.format(e))
            if i == 0:
                trainset += randomEpisode(game)
            else:
                trainset += executeEpisode(game, nnet)        
        for _ in range(EPOCHS):
            trainNNet(trainset, nnet)
        nnet.save('models/Gomoku_{}x{}-{}.txt'.format(game.size, game.size, game.win))
        np.save('selfplay/selfplay_set', trainset)
#        if i == int((1/3)*numIters):
#            nnet.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(lr=0.01),metrics=['accuracy'])
#        if i == int((2/3)*numIters):
#            nnet.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(lr=0.001),metrics=['accuracy'])
def trainNNet(transition, net):
    if len(transition) < minibatch_size:
        return
    X = []
    y1 = []
    y2 = []
    minibatch = random.sample(transition, minibatch_size)
    sample = random.sample(minibatch, 1)
    for s, pi, z in minibatch:
        X.append(s)
        y1.append(np.array(pi))
        y2.append(np.array([z]))
    print(X[0])
    print(y1[0])
    print(y2[0])
    net.fit(np.array(X), [np.array(y1), np.array(y2)],batch_size=minibatch_size)

def randomEpisode(game):
    be = []
    we = []
    game.reset()
    bw, ww, draw = False, False, False
    while not (bw or ww or draw):
        ls = np.random.uniform(low=0,high=1,size=game.size**2)
        pi = calPI(ls, 0)
        a = np.argmax(pi)
        if game.t == 0:
            syms = game.getSyms(game.b, game.t, pi)
            be += syms
        else:
            syms = game.getSyms(game.b, game.t, pi)
            we += syms        
        bw, ww, draw = game.ai_step(a)
        if bw:
            for i in be:
                i[2] = 1
            for i in we:
                i[2] = -1
            print('bw')
            game.render()
            return be+we
        elif ww:
            for i in be:
                i[2] = -1
            for i in we:
                i[2] = 1
            print('ww')
            game.render()
            return be+we
        elif draw:
            for i in be:
                i[2] = 0
            for i in we:
                i[2] = 0
            print('draw')
            game.render()
            return be+we
    
def executeEpisode(game, nnet):
    be = []
    we = []
    game.reset()
    mcts = MCTS()
    bw, ww, draw = False, False, False
    while not (bw or ww):
        s = game.get_state()
        for _ in range(numMCTSSims):
            mcts.search(game, nnet)
        N = mcts.N[str(game.b)]
        if game.c < don:
            pi = calPI(N, 1)
            prob = np.random.multinomial(len(pi), pi)
        else:
            pi = calPI(N, float('inf'))
            prob = pi
        
        if game.t == 0:
            syms = game.getSyms(game.b, game.t, pi)
            be += syms
        else:
            syms = game.getSyms(game.b, game.t, pi)
            we += syms

        a = np.argmax(prob)
        bw, ww, draw = game.ai_step(a)
        if bw:
            for i in be:
                i[2] = 1
            for i in we:
                i[2] = -1
            print('bw')
            game.render()
            return be+we
        elif ww:
            for i in be:
                i[2] = -1
            for i in we:
                i[2] = 1
            print('ww')
            game.render()
            return be+we
        elif draw:
            for i in be:
                i[2] = 0
            for i in we:
                i[2] = 0
            print('draw')
            game.render()
            return be+we
        game.render()

if __name__ == '__main__':
    env = Gomoku()
    PolicyIterSP(env)
