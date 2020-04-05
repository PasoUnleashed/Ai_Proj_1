import games
import checkers
import leaderboard
import struct
import keras
def load_agent_dna(filename,id_,gene_size):
    agents_file = '%s_pop.bin' %filename
    segment_size = (((gene_size+3)*4)+12)+1
    with open(agents_file,'rb') as f:
        binary = f.read(4)
        while(binary):
            b = struct.unpack('>l',binary)[0]
            if(b==id_):
                al =f.read(segment_size-4)
                fmt = '>fllff?'+('f'*(gene_size))
                al = struct.unpack(fmt,al)
                return al[5::]
            else:
                f.seek(segment_size-4,1)
            binary=f.read(4)
def load_model(filename,id_,gene_size):
    dna = load_agent_dna(filename,id_,gene_size)
    m = leaderboard.simple_model()
    w = leaderboard.dna_to_weights(dna,m)
    m.set_weights(w)
    return m
class Player(games.AlphaBetaPlayer):
    def __init__(self, id_,model_id,outgoing,incoming,model = None):
        super().__init__(id_,self,2)
        self.model_id = model_id
        self.outgoing=outgoing
        self.incoming = incoming
        self.model=model
    def __call__(self,state,id_):
        fact = 1
        if(self.id==2):
            fact=-1
        return fact*self.model.predict(leaderboard.state_to_vector(state))[0][0]
class Client:
    def __init__(self,id_,outgoing,incoming):
        self.id=id_
        self.outgoing = outgoing
        self.incoming = incoming
        self.run()
    def run(self):
        self.outgoing.put(('ready' ,None,None))
        while(True):
            while(self.incoming.empty()):
                pass
            message = self.incoming.get()
            flag = message[0]
            if(flag=='T'):
                exit(0)
            elif(flag == 'gs'):
                self.gene_size = message[1]
            elif(flag == 'lb'):
                self.leaderboard = message[1]
            elif(flag=='play'):
                p1_id=message[1]
                p2_id=message[2]
                p1p = Player(1,p1_id,self.outgoing,self.incoming,load_model(self.leaderboard,p1_id,self.gene_size))
                p2p = Player(2,p2_id,self.outgoing,self.incoming,load_model(self.leaderboard,p2_id,self.gene_size))
                self.outgoing.put(('started',))
                res = self.play_match(p1p,p2p)
                self.outgoing.put(('result',res))
    def play_match(self,player_1,player_2):
        game =checkers.Checkers(player_1,player_2,max_turns=40)        
        res = game.run_to_completion(max_turns=40)
        return res