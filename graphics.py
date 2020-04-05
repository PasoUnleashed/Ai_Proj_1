import pygame
import checkers
import games
import leaderboard

size = 800
cell_size = size//8
def draw_state(game,state,highlighted_pieces = {},highlighted_squares = {}):
        game.screen.fill((255,254,200))
        for i in range(8):
            for j in range(8):        
                if((j%2 == 0 and i % 2==0) or (j%2 == 1 and i % 2==1)):
                    pygame.draw.rect(game.screen,(125,125,125),(i*cell_size,j*cell_size,cell_size,cell_size))
                if((i,j) in highlighted_squares):
                        pygame.draw.rect(game.screen,highlighted_squares[(i,j)],(i*cell_size,j*cell_size,cell_size,cell_size),8)    
                piece = state.get_square_player(j,i)
                if(piece==1):
                    pygame.draw.circle(game.screen, (32,70,40),((i*cell_size)+(cell_size//2),(j*cell_size)+(cell_size//2)),int(cell_size//2.2))
                    if((i,j) in highlighted_pieces):
                        pygame.draw.circle(game.screen, highlighted_pieces[(i,j)],((i*cell_size)+(cell_size//2),(j*cell_size)+(cell_size//2)),int(cell_size//2.1),8)
                    if(state.is_king(j,i)):
                        pygame.draw.circle(game.screen, (255,0,0),((i*cell_size)+(cell_size//2),(j*cell_size)+(cell_size//2)),int(cell_size//4))
                elif(piece==2):
                    pygame.draw.circle(game.screen, (95,0,0),((i*cell_size)+(cell_size//2),(j*cell_size)+(cell_size//2)),int(cell_size//2.2))
                    if((i,j) in highlighted_pieces):
                        pygame.draw.circle(game.screen, highlighted_pieces[(i,j)],((i*cell_size)+(cell_size//2),(j*cell_size)+(cell_size//2)),int(cell_size//2.1),8)
                    if(state.is_king(j,i)):
                        pygame.draw.circle(game.screen, (255,0,0),((i*cell_size)+(cell_size//2),(j*cell_size)+(cell_size//2)),int(cell_size//4))
class Human_Player(games.Player):
    def __init__(self,_id):
        super().__init__(_id)
    def play(self,game,state):
        highlighted_sqaures ={}
        highlighted_pieces = {}
        highlighted_s_moves = {}

        phase = 'selection'
        while True:
            if(phase=='selection'):
                selected_piece = (-1,-1)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        crashed = True
                        exit()
                    elif event.type == 6:
                        mx,my = pygame.mouse.get_pos()
                        print(mx,my)
                        selected_piece = (mx//cell_size,my//cell_size)
                        print(selected_piece)
                        highlighted_pieces = {selected_piece:(80,65,115)}
                        break
                y,x = selected_piece
                moves,isjump  = state.get_possible_states_from(selected_piece[1],selected_piece[0])
                changed = []
                key = state.board[x][y]
                amoves = state.get_successors()
                for mv in moves:
                    ex=False
                    for i in amoves:
                        if(i.board == mv.board):
                            ex =True
                    if(not ex):
                        continue
                    for i in range(x-2,x+3,1):
                        for j in range(y-2,y+3,1):
                            if(i<8 and i>=0 and j<8 and j>=0):
                                if(state.get_square_player(i,j) != self.id and mv.get_square_player(i,j) == self.id):
                                    highlighted_sqaures[(j,i)] = (80,65,115)
                                    highlighted_s_moves[(j,i)] = mv
                if(highlighted_pieces!={}):
                    phase ='action'
            elif phase == 'action':
                selected_square = (-1,-1)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        crashed = True
                        exit()
                    elif event.type == 6:
                        print('Click')
                        mx,my = pygame.mouse.get_pos()
                        print(mx,my)
                        selected_square = (mx//cell_size,my//cell_size)
                        print(selected_square,'sq')
                        break
                if(selected_square!=(-1,-1)):
                    if(selected_square in highlighted_s_moves):
                        return highlighted_s_moves[selected_square]
                    else:
                        phase='selection'
                        highlighted_pieces={}
                        highlighted_sqaures={}
                        highlighted_s_moves={}
            draw_state(game,state,highlighted_pieces=highlighted_pieces,highlighted_squares=highlighted_sqaures)
            pygame.display.flip()

class GraphicalCheckers(checkers.Checkers):
    def __init__(self,p1,p2):
        super().__init__(p1,p2) 
        self.on_turn.add(self.turn_event_handler)
        self.click_between = True
        if(isinstance(self.p1,Human_Player) or isinstance(self.p2,Human_Player)):
            self.click_between = False
            
    def run_to_completion(self,max_turns = -1):
        pygame.init()
        self.screen = pygame.display.set_mode([size,size])
        self.turn_event_handler(self)
        return super().run_to_completion(max_turns=max_turns)
    
    def turn_event_handler(self,game):
        print('called')
        running= True
        if(not self.click_between):
            return
        while(running):
            draw_state(self,self.state)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                if(event.type == 6) and self.click_between:
                    running=False
                break
                
l = leaderboard.Leaderboard('test',20,save_player_every=2,save_board_every=20)
p=l[1]
p.id = 1
p.depth = 5
#g = GraphicalCheckers(l[1],p)

g = GraphicalCheckers(p,Human_Player(2))
print(g.run_to_completion())
#g.run_to_completion()
#g = GraphicalCheckers(Human_Player(1),Human_Player(2))



    