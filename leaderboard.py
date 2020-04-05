from darwin import darwin
import sys
import os
import checkers
import games
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import predection_server
import keras
from keras.layers import Dense,Input,Conv2D,MaxPool2D
from keras.models import Sequential

sys.stderr = stderr
import numpy as np
import random
import gameplayer
import threading
import tensorflow as tf

import time
from multiprocessing import Pool,Queue
import multiprocessing as mp
from keras import backend as K
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 30, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s%s' % (prefix, bar, percent, suffix,printEnd))
    # Print New Line on Complete
    if iteration == total: 
        print()
def elo(a,b,a_score):
    e1 = 10**(a/400)
    e2 = 10**(b/400)
    k=20
    return k*((a_score)-(e1/(e1+e2))),k*((1-a_score)-(e2/(e2+e1)))    
def prod(x):
    r = 1
    for i in x:
        r*=i
    return r
def cal_model_wlen(model):
    w = model.get_weights()
    s=0
    for i in w:
        s+=prod(i.shape)
    return s

def simple_model():
    inp1 = Input(shape= (8,8,1))
    inp2 = Input(shape= (1,))
    x = Dense(1)(inp2)
    c = Conv2D(20,3,input_shape=(8,8),activation='sigmoid')(inp1)
    c = MaxPool2D(pool_size=(6,6))(c)
    c = keras.layers.Flatten()(c)
    c = keras.layers.Concatenate()([c,x])
    c = Dense(128,activation='relu')(c)
    o = Dense(1,activation='sigmoid')(c)
    model = keras.Model(inputs=[inp1,inp2],outputs=o)
    return model
def dna_to_weights(dna,model):
        weights = model.get_weights()
        dna = list(dna)
        splt = []
        cur = 0 
        for i in weights:
            splt.append(np.array(dna[cur:cur+prod(i.shape)]))
            cur+=prod(i.shape)
            splt[-1] = splt[-1].reshape(i.shape) 
        return splt
def create_model_for(agent):
    model = simple_model()
    weights = dna_to_weights(agent.dna,model)
    model.set_weights(weights)
    model._make_predict_function()
    return model
def state_to_vector(state):
    rn =[]
    for i in state.board:
            a = np.array(i)
            rn.append((a+2)/4)
    rn = np.array(rn)
    ret = []
    ret.append(rn)
    ret.append(np.array([state.player-1]))
    ret[0]= ret[0].reshape((1,8,8,1))
    ret[1] = ret[1].reshape((1,1))
    return ret
class KerasHeuristicPlayer(games.AlphaBetaPlayer):
    def __init__(self,_id,model=None,agent=None):
        super().__init__(_id,self,2)
        self.model = model
        self.agent = agent
        if(not model):
            self.model = create_model_for(agent)
    def __call__(self,state,_id):
        model= self.model
        ret = state_to_vector(state)
        fact = 1
        if(_id==2):
            fact=-1
        return fact*model.predict(ret)[0][0]
    
    def get_rating(self):
        return abs(self.agent.evaluation)

class Leaderboard:
    gamecache = {}
    def __init__(self,name,size,save_player_every=3,save_board_every=100,thread_count = 5,ow=False,max_saved=20):
        self.status = 'stopped'
        self.save_player_every = save_player_every
        self.save_board_every = save_board_every
        self.name = name
        self.size=size
        self.model = self.create_darwin_model(self.size)
        self.mname ='%s_dm'%name
        self.max_saved = max_saved
        self.avg = 800
        self.thread_count=thread_count
        if(ow):
            print('Creating New Model (Overwrite on)')
            [i.evaluate(800) for i in self.model.population]
            self.model.save(self.mname)
            print('Model Created')
        else:
            try:
                print('Loading model')
                self.model.load(self.mname)
                print('Model loaded')
            except Exception as e:
                print('Creating new model..')
                [i.evaluate(800) for i in self.model.population]
                #self.model.save(self.mname)  
                raise Exception('model not found')
                print('Model Created') 
        print('Creating prediction server')
        self.prediction_server = predection_server.Server(thread_count,simple_model,self.mname)
        self.prediction_server.on_match_complete.add(self.eval_match)
        
        print('Server created')
    def create_darwin_model(self,count):
        breeder = darwin.WeightedSelectionBreeder(model = None,top_echelon = 0.4)
        merger = darwin.MomentumMerger(model = None,crossover = 0.5,mutation_chance = 0.5,mutation_rate =1,max_mutations =0,decay_every=1)
        return darwin.StaticEvaluationModel(gene_size=cal_model_wlen(simple_model()),pop_size=count,breeder =breeder,merger = merger,default_constraint=darwin.Constraint(0,1,8),constraints={},on_agent_added=lambda x: [i.evaluate(max(800,self.avg)) for i in x])
    def __getitem__(self, key):
        x = [i for i in self.get_all_players()]
        x.sort(key=lambda m:m.evaluation,reverse = True)
        return KerasHeuristicPlayer(1,agent = x[key])
    
    def train_for(self,generations = 20,pool_size=3):
        self.prediction_server.start()
        for i in range(generations):
            if(i>0 and i%self.save_board_every==0):
                print('Saving... ',end='')
                self.model.save(self.mname)
                print('Saved.')
            self.do_gen(pool_size=pool_size)
            self.print_leaderboard(max_saved=self.max_saved)
            if(((self.model.tdelta-1)%self.save_player_every)==0):
                self.model.get_best().save() 
    def save(self):
        self.model.save(self.mname)
    def eval_match(self,match,res):
        if(res=='p1win'):
            p1d,p2d = elo(match[0].evaluation,match[1].evaluation,1)
        elif(res=='p2win'):
            p1d,p2d = elo(match[0].evaluation,match[1].evaluation,0)
        else:
            p1d,p2d = elo(match[0].evaluation,match[1].evaluation,0.5)
        match[0].evaluate((match[0].evaluation+p1d))
        match[1].evaluate((match[1].evaluation+p2d))
    def evaluate_population(self,pop=None,pool_size=3):
        self.model.save(self.mname)
        for i in range(2):
            matches = self.create_match_list(pool_size,players=self.get_all_players(max_saved=self.max_saved))
            self.prediction_server.play_matches(matches)
        self.avg = self.model.get_average_eval(pop=self.get_all_players())
        self.model.save(self.mname)
    def do_gen(self,pool_size=3):
           
        self.evaluate_population(pool_size=pool_size)
        self.model.TimeStep()
    def create_match_list(self,pool_size,players=None):
        if(not players):
            players = self.get_all_players(max_saved=self.max_saved)
        pools = []
        players.sort(key=lambda x: x.evaluation,reverse=True)
        i=0
        printProgressBar(0,len(players)//pool_size,prefix="Creating pools".ljust(17))
        while(i<len(players)):
            
            if(i==0 and len(players)%pool_size <3):
                pools.append(players[i:min(i+pool_size+(len(players)%pool_size)+1,len(players))])
                i+=(len(players)%pool_size)
            else:
                pools.append(players[i:min(i+pool_size,len(players))])
            i+=pool_size
            printProgressBar(round((i+1)/pool_size),len(players)//pool_size,prefix="Creating pools..".ljust(17),suffix="%d/%d"%(i//pool_size,len(players)//pool_size))
        matches = []
        total = 0
        for i in pools:
            total+=len(i)*((len(i))-1)
        printProgressBar(0,total,prefix="Creating Matches".ljust(17))
        cur = 0
        for i in pools:
            for j in range(len(i)-1):
                for l in i[j+1::]:
                    matches.append((i[j],l))
                    matches.append((l,i[j]))
                    cur+=2
            printProgressBar(cur,total,prefix="Creating Matches".ljust(17),suffix="%d/%d"%(cur,total))
        return matches
    
    def get_appropriate_opponent(self,rating,players=None):
        
        if(not players):
            players = self.get_all_players()
        max_dist = float('-inf')
        for i in players:
            max_dist = max(max_dist,abs(rating-i.evaluation))
        if(max_dist == 0):
            return random.choice(players)
        weights = [(abs(rating-i.evaluation)/max_dist)*len(players) for i in players]
        selct = random.choices(players,weights=weights,k=1)[0]
        return selct

    def get_all_players(self,max_saved=-1):
        ins =[i for i in self.model.population]
        saved = [i for i in self.model.saved_population]
        s2 = [i for i in saved]
        for i in s2:
            for j in ins:
                if(i.id==j.id):
                    saved.remove(j)
        saved.sort(key = lambda x: x.evaluation, reverse = True)
        if(max_saved!=-1):
            saved = saved[0:min(len(saved),self.max_saved)]
        return ins+saved
    def find_pair(self):
        pass
    def print_leaderboard(self,max_saved=-1):
        header = "Generation:%d\n|%s|%s|%s|" % (self.model.tdelta,"#".ljust(4),"ID".ljust(8),"Rating".ljust(8))
        print(header)

        p =self.get_all_players(max_saved)
        p.sort(key = lambda x: x.evaluation, reverse = True)
        for i,p in enumerate(p):
            ad=""
            if(p in self.model.saved_population):
                ad="*"
            print("|%s|%s|%s|" % ((str(i+1)+ad).ljust(4),str(p.id).ljust(8),(str(round(p.evaluation,2)).ljust(8))))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
if __name__=='__main__':
    print(cal_model_wlen(simple_model()))
    mp.set_start_method('spawn')
    l = Leaderboard('test',50,save_player_every=20,save_board_every=1,max_saved=10,thread_count =7,ow=False)
    l.print_leaderboard()
    l.train_for(1000,pool_size = 5)
    l.save()