from games import Game, State,RandomPlayer,AlphaBetaPlayer
def switch_p(p):
    if(p==1):
        return 2
    elif(p==2):
        return 1
    return 0
class Checkers(Game):
    def __init__(self,p1,p2,max_turns=-1):
        super().__init__(p1,p2,max_turns=max_turns)
    def get_start_state(self):
        board = [[0 for j in range(8)] for i in range(8)]
        for i in range(len(board)):
            for j in range(len(board[i])):
                if(i<3 and ((j%2==0 and (i+1) %2==0) or (j%2==1 and (i+1) %2==1 ))):
                    board[i][j] = -1
                if(i>=5 and ((j%2==0 and (i+1) %2==0) or (j%2==1 and (i+1) %2==1 ))):
                    board[i][j] = 1
        return CheckersState(1,board)
    def get_status(self):
        return self.state._evaluate_status()
class CheckersState(State):
    def __init__(self,player,board,islocked=False,lockx=-1,locky=-1,evaluate = True):
        super().__init__(player)
        self.board =board
        self.islocked = islocked
        self.lockx=lockx
        self.locky=locky
        self.check_upgrade()
    def get_status(self):
        return self._evaluate_status()
    def _evaluate_status(self):
        status = "running"
        p1p,p2p = self.count_pieces()

        has = False
        for i in range(8):
            for j in range(8):
                if(self.get_possible_states_from(i,j)[0]):
                    has = True
                    break
        if(not has):
            return "p%dwin" % switch_p(self.player)
        if(p1p==0):
            status = "p2win"
        if(p2p==0):
            status = "p1win"
        if(p2p==p1p and p1p == 1):
            status = 'draw'
        self.status=status
        return status
    def get_successors(self):
        if(self.islocked):
            return self.get_possible_states_from(self.lockx,self.locky)[0]
        eats = []
        moves = []
        for i in range(8):
            for j in range(8):
                m,t = self.get_possible_states_from(i,j)
                if(t):
                    eats+=m
                elif (m):
                    moves+=m
        if(len(eats)==0):
            return moves
        return eats
    def get_dirs_for(self,x,y):
        dirs = [(-1,-1),(-1,1),(1,-1),(1,1)]
        if(self.get_square_player(x,y)==1 and self.is_pawn(x,y)):
            dirs.remove((1,1))
            dirs.remove((1,-1))
        elif(self.get_square_player(x,y)==2 and self.is_pawn(x,y)):
            dirs.remove((-1,1))
            dirs.remove((-1,-1))
        return dirs

    def get_possible_states_from(self,x,y):
        dirs = self.get_dirs_for(x,y)
        moves = []
        eats = []
        for dx,dy in dirs:
            if(self.is_in_board(x+dx,y+dy)):
                m,t = self.get_states_from_in_direction(x,y,dx,dy)
                if(not m):
                    continue
                if(not t):
                    moves.append(m)
                else:
                    eats.append(m)
        if(len(eats)>0):
            return eats,True
        return moves,False
    def can_eat_from(self,x,y,state=None):
            if(state == None):
                state=self
            dirs = state.get_dirs_for(x,y)
            ret = []
            for dx,dy in dirs:
                dx2 = dx*2
                dy2 = dy*2
                if(not state.is_in_board(x+dx2,y+dy2)):
                    continue
                if(state.get_square_player(x+dx,y+dy)==switch_p(self.player)):
                    if(state.get_square_player(x+dx2,y+dy2)==0):
                        ret.append((dx,dy))
            return ret
    def get_states_from_in_direction(self,x,y,dx,dy):
        # Check move
        
        if(not self.is_in_board(x+dx,y+dy)):
            return None,None
        p = self.get_square_player(x+dx,y+dy)
        if(self.get_square_player(x,y)!=self.player):
            return None,None
        if(p==0):
            ns = self.copy()
            ns.board[x+dx][y+dy] = ns.board[x][y]
            ns.board[x][y] = 0
            ns.player = switch_p(ns.player)
            ns.check_upgrade()
            return ns,False # False== Move
        else:
            caneatlist = self.can_eat_from(x,y)
            if(not caneatlist):
                return None,None
            if((not (dx,dy) in caneatlist) or (not self.is_in_board(x+(dx*2),y+(dy*2)))):
                return None,None
            ns = self.copy()
            ns.board[x+dx][y+dy]=0
            ns.board[x+(dx*2)][y+(dy*2)]=ns.board[x][y]
            ns.board[x][y] = 0
            if(self.can_eat_from(x+(dx*2),y+(2*dy),ns)):
                ns.islocked=True
                ns.lockx = x+(dx*2)
                ns.locky = y+(dy*2)
            else:
                ns.player = switch_p(ns.player)
                ns.islocked = False
            ns.check_upgrade()
            return ns, True
        return None,None

    def is_in_board(self,x,y):
        if(x>=0 and y>=0 and x<=7 and y <=7):
            return True
        return False
    def get_square_player(self,x,y):
        if(not self.is_in_board(x,y)):
            return -1
        if(self.board[x][y]<0):
            return 2
        elif(self.board[x][y]>0):
            return 1
        return 0
    def is_king(self,x,y):
        return abs(self.board[x][y])==2
    def is_pawn(self,x,y):
        return abs(self.board[x][y])==1
    def check_upgrade(self):
        for i in range(8):
            if(self.board[0][i]==1):
                self.board[0][i]=2
            if(self.board[7][i]==-1):
                self.board[7][i]=-2
    def count_pieces(self):
        p1c = 0
        p2c = 0
        for i in self.board:
            for j in i:
                if(j>0):
                    p1c+=1
                elif(j<0):
                    p2c+=1
        return p1c,p2c
    def copy(self):
        return CheckersState(self.player,[[self.board[i][j] for j in range(8)] for i in range(8)],self.islocked,self.lockx,self.locky)
    def print_board(self):
        for j in range(8):
            string = ""
            for i in range(8):
                player = self.get_square_player(j,i)
                if(player == 1):
                    if(self.is_pawn(j,i)):
                        string+='X'
                    else:
                        string+='K'
                elif(player == 2):
                    if(self.is_pawn(j,i)):
                        string+='O'
                    else:
                        string+='Q'
                else:
                    string+=' '
                if(i<7):
                    string+='|'
            print(j,string)
        print()
def simph(state,id_):
    stat = state.get_status()
    if(stat=='p1win'):
        if( id_==1):
            return float('inf')
        else:
            return float('-inf')
    elif(stat=='p2win'):
        if( id_==1):
            return float('-inf')
        else:
            return float('inf')
    total=0
    for i in range(8):
        for j in range(8):
            if(state.get_square_player(i,j)==id_):
                if(state.is_pawn(i,j)):
                    total+=3
                else:
                    total+=5
            elif(state.get_square_player(i,j)==switch_p(id_)):
                if(state.is_pawn(i,j)):
                    total-=3
                else:
                    total-=5
    return total
