import math
import numpy as np
import copy
from Gomoku_v8 import Gomoku

class MCTS:
    def __init__(self):
        self.Q = {}
        self.P = {}
        self.N = {}
        self.cpuct = 5
        self.game = Gomoku()

    def search(self, og, net):
        
        rb = copy.deepcopy(og.b)
        rt = copy.copy(og.t)
        rc = copy.copy(og.c)
        self.game.set(rb,rt,rc)

        state = self.game.get_state()


        if str(rb) not in self.P:
            p, v = net.predict(np.array([state]))
            p = p[0]
            v = v[0][0]
            self.P[str(rb)] = p
            self.Q[str(rb)] = [0 for _ in range(self.game.size**2)]
            self.N[str(rb)] = [0 for _ in range(self.game.size**2)]
            return -v
        
        max_u = -float('inf')
        best_a = -1
        for a in range(self.game.size**2):
            if self.game.legal_check(a):
                u = self.Q[str(rb)][a] + self.cpuct*self.P[str(rb)][a]*math.sqrt(sum(self.N[str(rb)]))/(1+self.N[str(rb)][a])
                if u > max_u:
                    max_u = u
                    best_a = a
        a = best_a

        bw, ww, draw = self.game.ai_step(a)
        if bw or ww: 
            v = 1
            self.Q[str(rb)][a] = (self.N[str(rb)][a]*self.Q[str(rb)][a]+v)/(self.N[str(rb)][a]+1)
            self.N[str(rb)][a] += 1
            return -v

        elif draw:
            v = 0
            self.Q[str(rb)][a] = (self.N[str(rb)][a]*self.Q[str(rb)][a]+v)/(self.N[str(rb)][a]+1)
            self.N[str(rb)][a] += 1
            return v

        else:
            v = self.search(self.game, net)

            self.Q[str(rb)][a] = (self.N[str(rb)][a]*self.Q[str(rb)][a]+v)/(self.N[str(rb)][a]+1)
            self.N[str(rb)][a] += 1
        return -v

