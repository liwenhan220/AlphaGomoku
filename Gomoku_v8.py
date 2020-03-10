import math
import copy
import numpy as np

class Gomoku:
    def __init__(self):
        self.size = 6
        self.win = 4
        self.t = 0
        self.b = [['.' for _ in range(self.size)] for _ in range(self.size)]
        self.c = 0
        
    def set(self, nb, nt, nc):
        self.b = copy.deepcopy(nb)
        self.t = copy.copy(nt)
        self.c = copy.copy(nc)
        
    def reset(self):
        self.t = 0
        self.b = [['.' for _ in range(self.size)] for _ in range(self.size)]
        self.c = 0
   
    def getSyms(self, rb, rt, pi):
        syms = []
        reshaped_pi = [[0 for _ in range(self.size)] for _ in range(self.size)]
        for n in range(len(pi)):
            x = math.floor(n/self.size)
            y = n % self.size
            reshaped_pi[x][y] = pi[n]

        for i in range(5):
            for j in [True, False]:
                newB = np.rot90(rb, i)
                newPi = np.rot90(reshaped_pi, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                pi2 = [0 for _ in range(self.size**2)]
                for x in range(len(reshaped_pi)):
                    for y in range(len(reshaped_pi[x])):
                        pi2[x*self.size+y] = reshaped_pi[x][y]
                s = self.get_sstate(newB, rt)
                syms.append([s, pi2, None])
        return syms       

    def check_draw(self):
        if self.c >= self.size**2:            
            return True
        return False

    def legal_check(self, n):
        x = math.floor(n/self.size)
        y = n % self.size
        if self.b[x][y] == '.':
            return True
        return False

    def check_win(self, x, y):
        
        if self.t == 1:
            b_succ = any([self.backward1(x,y,0)+self.forward1(x,y,0)>self.win,
                          self.backward2(x,y,0)+self.forward2(x,y,0)>self.win,
                          self.backward3(x,y,0)+self.forward3(x,y,0)>self.win,
                          self.backward4(x,y,0)+self.forward4(x,y,0)>self.win])
            return b_succ, False, self.check_draw()
        else:
            w_succ = any([self.backward1(x,y,1)+self.forward1(x,y,1)>self.win,
                          self.backward2(x,y,1)+self.forward2(x,y,1)>self.win,
                          self.backward3(x,y,1)+self.forward3(x,y,1)>self.win,
                          self.backward4(x,y,1)+self.forward4(x,y,1)>self.win])
            return False, w_succ, self.check_draw()
    
    def backward1(self, x, y, t):
        if t == 0:
            counter = 0
            for i in range(5):
                if x-i < 0 or y-i < 0 or self.b[x-i][y-i] != 'o':
                    break
                
                counter += 1
            return counter
        else:
            counter = 0
            for i in range(5):
                if x-i < 0 or y-i < 0 or self.b[x-i][y-i] != 'x':
                    break
                
                counter += 1
            return counter
            
    # Horizontal backwards
    def backward2(self, x, y, t):
        if t == 0:
            counter = 0
            for i in range(5):
                if x-i < 0 or self.b[x-i][y] != 'o':
                    break
                
                counter += 1
            return counter
        else:
            counter = 0
            for i in range(5):
                if x-i < 0 or self.b[x-i][y] != 'x':
                    break
                
                counter += 1
            return counter
        
    # vertical backwards
    def backward3(self, x, y, t):
        if t == 0:
            counter = 0
            for i in range(5):
                if y-i < 0 or self.b[x][y-i] != 'o':
                    break
                
                counter += 1
            return counter
        else:
            counter = 0
            for i in range(5):
                if y-i < 0 or self.b[x][y-i] != 'x':
                    break
                
                counter += 1
            return counter

    #
    def backward4(self, x, y, t):
        if t == 0:
            counter = 0
            for i in range(5):
                if x-i < 0 or y+i >= len(self.b) or self.b[x-i][y+i] != 'o':
                    break
                
                counter += 1
            return counter
        else:
            counter = 0
            for i in range(5):
                if x-i < 0 or y+i >= len(self.b) or self.b[x-i][y+i] != 'x':
                    break
                
                counter += 1
            return counter

    def forward1(self, x, y, t):
        if t == 0:
            counter = 0
            for i in range(5):
                if x+i >= len(self.b) or y+i >= len(self.b) or self.b[x+i][y+i] != 'o':
                    break
                
                counter += 1
            return counter
        else:
            counter = 0
            for i in range(5):
                if x+i >= len(self.b) or y+i >= len(self.b) or self.b[x+i][y+i] != 'x':
                    break
                
                counter += 1
            return counter

    def forward2(self, x, y, t):
        if t == 0:
            counter = 0
            for i in range(5):
                if x+i >= len(self.b) or self.b[x+i][y] != 'o':
                    break
                
                counter += 1
            return counter
        else:
            counter = 0
            for i in range(5):
                if x+i >= len(self.b) or self.b[x+i][y] != 'x':
                    break
                
                counter += 1
            return counter
        

    def forward3(self, x, y, t):
        if t == 0:
            counter = 0
            for i in range(5):
                if y+i >= len(self.b) or self.b[x][y+i] != 'o':
                    break
                
                counter += 1
            return counter
        else:
            counter = 0
            for i in range(5):
                if y+i >= len(self.b) or self.b[x][y+i] != 'x':
                    break
                
                counter += 1
            return counter

    def forward4(self, x, y, t):
        if t == 0:
            counter = 0
            for i in range(5):
                if x+i >= len(self.b) or y-i < 0 or self.b[x+i][y-i] != 'o':
                    break
                
                counter += 1
            return counter
        else:
            counter = 0
            for i in range(5):
                if x+i >= len(self.b) or y-i < 0 or self.b[x+i][y-i] != 'x':
                    break
                
                counter += 1
            return counter
        
    def render(self):
        for i in self.b:            
            print(*i)
        for _ in range(5):
            print('')

    def step(self, x, y):
        if self.b[x][y] != '.':
            return self.check_win(x,y)
        self.c += 1
        if self.t == 0:
            self.b[x][y] = 'o'
            self.t = 1
            
        else:
            self.b[x][y] = 'x'
            self.t = 0
        return self.check_win(x,y)
    
    def ai_step(self, n):
        x = math.floor(n/self.size)
        y = n % self.size
        if self.b[x][y] != '.':
            return self.check_win(x,y)
        self.c += 1
        if self.t == 0:
            self.b[x][y] = 'o'
            self.t = 1
            
        else:
            self.b[x][y] = 'x'
            self.t = 0
        return self.check_win(x,y)

    def justify(self,ls):
        ls = list(ls)
        for x in range(len(self.b)):
            for y in range(len(self.b[x])):
                if self.b[x][y] != '.':
                    ls[x*self.size+y] = float('-inf')
        return ls
    
    def get_state(self):
        state = [[[0,0,self.t] for _ in range(self.size)] for _ in range(self.size)]
        for i in range(len(self.b)):
            for ii in range(len(self.b[i])):
                if self.b[i][ii] == 'o':
                    state[i][ii][0] = 1
                elif self.b[i][ii] == 'x':
                    state[i][ii][1] = 1
        return np.array(state).astype(np.uint8)

    def get_sstate(self, fb, t):
        state = [[[0,0,t] for _ in range(self.size)] for _ in range(self.size)]
        for i in range(len(fb)):
            for ii in range(len(fb[i])):
                if fb[i][ii] == 'o':
                    state[i][ii][0] = 1
                elif fb[i][ii] == 'x':
                    state[i][ii][1] = 1
        return np.array(state).astype(np.uint8)
