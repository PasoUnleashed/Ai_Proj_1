from abc import ABC,abstractmethod
import random
class Game(ABC):
    on_turn = set()
    def __init__(self,p1,p2,max_turns=-1):
        super().__init__()
        self.p1=p1
        self.p2=p2
        self.turns = 0
        self.state = self.get_start_state()
        self.max_turns=max_turns
    @abstractmethod
    def get_start_state(self):
        pass

    def get_status(self):
        return self.state.status
    def turn(self):
        if(self.state.player == 1):
            ret= self.p1.play(self,self.state)
            if(ret.player!=self.state.player):
                self.turns+=1
            return ret
        else:
            ret= self.p2.play(self,self.state)
            if(ret.player!=self.state.player):
                self.turns+=1
            return ret
        
    def one_turn(self):
        selection = self.turn()
        for i in self.on_turn:
            try: 
                i(self)
            except Exception as e:
                raise e
        self.state =selection
        if(self.turns>self.max_turns and self.max_turns!=-1):
            return 'draw'
    def run_to_completion(self,max_turns = -1):
        while(self.get_status()=='running' and not (self.turns>max_turns and max_turns!=-1)):
            self.one_turn()
        return self.get_status()
class Player(ABC):
    def __init__(self,id_):
        super().__init__()
        self.id = id_
    @abstractmethod
    def play(self, game,state):
        pass
class State(ABC):
    def __init__(self,player):
        super().__init__()
        self.player = player
        self.status = 'n/a'
    @abstractmethod
    def get_successors(self):
        pass
    @abstractmethod
    def get_status(self):
        pass
class RandomPlayer(Player):
    def __init__(self,id_):
        super().__init__(id_)
    def play(self,game,state):
        return random.choice(state.get_successors())
class AlphaBetaPlayer(Player):
    def __init__(self,id_,heuristic,depth):
        super().__init__(id_)
        self.heuristic = heuristic
        self.depth = depth
    def play(self,game,state):
        succ = state.get_successors()
        random.shuffle(succ)
        total_expanded= 0
        maxchild = 0
        maxchild_val,exp = self.minimax(succ[0],self.depth-1,float('-inf'),float('inf')) 
        total_expanded+=exp
        for i in range(1,len(succ)):
            evl,exp = self.minimax(succ[i],self.depth-1,float('-inf'),float('inf'))
            if(evl>maxchild_val):
                maxchild_val = evl
                maxchild = i
            total_expanded+=exp+1
        return succ[maxchild]
    def minimax(self,state,depth,alpha,beta):
        if(depth==0 or state.get_status()!='running'):
            return self.heuristic(state,self.id),1
        total_expanded= 0
        succ = state.get_successors()
        random.shuffle(succ)
        if self.id==state.player:
            maxEval = float('-inf')
            for i in succ:
                evalu,exp = self.minimax(i,depth-1,alpha,beta)
                total_expanded+=exp
                maxEval = max(maxEval,evalu)
                alpha = max(alpha,evalu)
                if(beta<=alpha):
                    break
            return maxEval,total_expanded+1
        else:
            minEval = float('inf')
            for i in succ:
                evalu,exp = self.minimax(i,depth-1,alpha,beta)
                minEval = min(minEval,evalu)
                beta = min(beta,evalu)
                total_expanded+=exp
                if(beta<=alpha):
                    break
                
            return minEval,total_expanded+1

        