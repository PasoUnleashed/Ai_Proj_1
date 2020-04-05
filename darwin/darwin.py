from abc import ABC, abstractmethod
import random
import numpy as np
import math
from numpy.random import choice
import struct
random.seed()
Agent_ids = 0
def merger_from_string(string):
    if(string in _mergers):
        return _mergers[string](None)
def breeder_from_string(string):
    if(string in _breeders):
        return _breeders[string](None)
class Constraint:
    def __init__(self,min_val,max_val,max_accuracy):
        self.min_val=min_val
        self.max_val=max_val
        self.max_accuracy =max_accuracy
        
    def apply(self,v):
        return min(max(v,self.min_val),self.max_val)
    def random(self,ranged=True):
        ret =0
        if(ranged):
            ret = self.min_val
        return ret + random.random()*(self.max_val-self.min_val)
    def __str__(self):
        return "%s,%s,%d" % (str(self.min_val),str(self.max_val),self.max_accuracy)
class Agent:
    def __init__(self,gene_size,dna=[],parent_evals=None,parenta_id=None,parentb_id =None,evaluation = 0,binary = None):
        global Agent_ids
        if(binary):
            self.gene_size = gene_size
            unpack = self.bin2array(binary)
            self.id = unpack[0]
            self.evaluation = unpack[1]
            if(unpack[2]!=-1):
                self.parenta_id = unpack[2]
                self.parentb_id = unpack[3]
                self.parent_evals = (unpack[4],unpack[5])
                Agent_ids = max(self.parenta_id,Agent_ids)
                Agent_ids = max(self.parentb_id,Agent_ids)
            else:
                self.parenta_id = None
                self.parentb_id = None
                self.parent_evals=(None,None)
            self.saved = unpack[6]
            Agent_ids = max(Agent_ids,self.id)
            self.dna = unpack[7::]
        else:
            self.saved=False
            self.id = Agent_ids
            self.gene_size = gene_size
            Agent_ids+=1
            self.dna = dna
            if not dna:
                self.dna = [0 for i in range(gene_size)]
            self.parenta_id = parenta_id
            self.parentb_id = parentb_id
            self.parent_evals = parent_evals
            self.evaluation = evaluation
    def evaluate(self, evaluation):
        self.evaluation =evaluation
    def get_bin(self):
        fmt = '>lfllff?'+('f'*(len(self.dna)))
        pa = -1
        pb = -1
        p1ev = -1
        p2ev = -1
        if(self.parenta_id):
            pa = self.parenta_id
            pb = self.parentb_id
            p1ev = self.parent_evals[0]
            p2ev = self.parent_evals[1]
        ret = struct.pack(fmt,*([self.id,self.evaluation,pa,pb,p1ev,p2ev,self.saved]+self.dna))
        return ret
    def bin2array(self,binary):
        fmt = '>lfllff?'+('f'*(self.gene_size))
        ret = [i for i in struct.unpack(fmt,binary)]
        return ret
    def save(self):
        self.saved=True
class Model(ABC):
    on_agents_removed = set()
    on_agents_added = set()
    on_time_step = set()
    saved_population = set()
    tdelta = 0
    def __init__(self,gene_size,pop_size,merger,breeder,default_constraint,constraints,on_agent_added = None):
        super().__init__()
        self.default_constraint =default_constraint
        self.constraints = constraints
        self.pop_size=pop_size
        self.gene_size = gene_size
        if(on_agent_added):
            self.on_agents_added.add(on_agent_added)
        if(isinstance(breeder,str)):
            if(breeder in _breeders):
                self.breeder = _breeders[breeder](self)
            else:
                raise Exception('Breeder not found',breeder)
        else:
            self.breeder = breeder
            breeder.set_model(self)
        if(isinstance(merger,str)):
            if(merger in _mergers):
                self.merger = _mergers[merger](self)
            else:
                raise Exception('Merger not found',merger)
        else:
            self.merger =merger
            self.merger.set_model(self)
        self.population = self._generate_random_pop(self.pop_size)
    def TimeStep(self):
        cgen = [i for i in self.population]
        self._timestep()
        news = [i for i in self.population if i not in cgen]
        rem = [i for i in cgen if i not in self.population]
        self._agents_added(news)
        self._agents_removed(rem)
        self.tdelta+=1
        for i in self.on_time_step:
            i(self.tdelta)
    @abstractmethod
    def _timestep(self):
        pass
    @abstractmethod
    def _load_conf(self,conf_string):
        pass
    @abstractmethod
    def _get_conf_string(self):
        pass
    def save(self,name):
        conf = self._get_conf_string()
        b_conf = self.breeder._get_conf_string()
        m_conf = self.merger._get_conf_string()
        with open('%s_conf.txt'%name,'w') as f:
            f.write(conf)
            f.write("\n")
            f.write(b_conf)
            f.write("\n")
            f.write(m_conf)
            f.write("\n")
        self.breeder._save(name)
        self.merger._save(name)
        self.save_pop(name)
    def save_pop(self,name):
        with open("%s_pop.bin" % name,'wb') as f:
            for i in self.population:
                f.write(i.get_bin())
            for i in self.saved_population:
                f.write(i.get_bin())
    def load_pop(self,name):
        with open('%s_pop.bin' % name,'rb') as f:
            genebsize = (((self.gene_size+3)*4)+12)+1
            binary = f.read(genebsize)
            cnt = 0
            while(binary):
                a = Agent(gene_size = self.gene_size,binary = binary)
                if(cnt<self.pop_size):
                    self.population.append(a)
                else:
                    self.saved_population.add(a)
                cnt+=1
                binary = f.read(genebsize)
    def load(self,name):
        try:
            self.population.clear()
            self.saved_population.clear()
            with open('%s_conf.txt' % name,'r') as f:
                lines = f.readlines()
                self._load_conf(lines[0])
                self.load_pop(name)
                self.breeder = breeder_from_string(lines[1].split(',')[0].strip())
                self.merger = merger_from_string(lines[2].split(',')[0].strip())
                self.breeder._load_conf(lines[1])
                self.merger._load_conf(lines[2])
                self.breeder._load(name)
                self.merger._load(name)
                self.merger.set_model(self)
                self.breeder.set_model(self)
        except Exception as e:
            print("Error loading model %s" % name)
            raise e
    
    def _generate_agent(self,dna = None,parent_a=None,parent_b=None):
        if(parent_a and parent_b):
            if not dna:
                dna = self.merger.gen_gene(self.gene_size,self.default_constraint,self.constraints)
            return Agent(self.gene_size,dna,(parent_a.evaluation,parent_b.evaluation),parent_a.id,parent_b.id)
        else:
            if(dna):
                return Agent(self.gene_size,dna =dna)
            else:
                return Agent(self.gene_size,dna =self.merger.gen_gene(self.gene_size,self.default_constraint,self.constraints))
    def _generate_random_pop(self,count):
        return [self._generate_agent() for i in range(count)]
    def get_best(self,count=1):
        if(count==1):
            maximum_v = float('-inf')
            maximum_i = 0
            for i in range(len(self.population)):
                if(self.population[i].evaluation>maximum_v):
                    maximum_v = self.population[i].evaluation
                    maximum_i = i
            return self.population[maximum_i]
        else:
            self.population.sort(key=lambda x: x.evaluation,reverse=True)
            return self.population[0:(min(count,len(self.population)))]
    def get_average_eval(self,pop=None):
        if(pop==None):
            pop = self.population
        total = 0
        for i in pop:
            total+=i.evaluation   
        return total/len(pop)
    def _agents_removed(self,agents):
        for i in agents:
            if(i.saved):
                self.saved_population.add(i)
        for i in self.on_agents_removed:
            try:
                i(agents)
            except:
                raise Exception("Error calling on_removed_event")
    def _agents_added(self,agents):
        for i in self.on_agents_added:
            try:
                i(agents)
            except Exception as e:
                raise e
    def get_constraint(self,i):
        if(i in self.constraints):
            return self.constraints[i]
        return self.default_constraint


class StaticEvaluationModel(Model):

    
    def __init__(self,pop_size=None,breeder=None,merger=None,gene_size=None,default_constraint = None,constraints={},on_agent_added=None,file=None):
        super().__init__(gene_size,pop_size,merger,breeder,default_constraint,constraints,on_agent_added=on_agent_added)
        
    def _timestep(self):
        self._move_next()
    def _move_next(self):
        npop,removed = self.breeder.next_generation(self.tdelta)
        self.population = npop
        self._agents_removed(removed)
    def _get_conf_string(self):
        ar = [self.pop_size,self.gene_size,self.tdelta,str(self.default_constraint)]+[str(i)+","+str(self.constraints[i]) for i in self.constraints]
        string = ""
        for i in range(len(ar)):
            string+=str(ar[i])
            if(i!=len(ar)-1):
                string+=","
        return string
    def _load_conf(self,conf_string):
        splt = conf_string.split(',')
        self.pop_size = int(splt[0])
        self.gene_size = int(splt[1])
        self.tdelta = int(splt[2])
        mv = float(splt[3])
        mxv = float(splt[4])
        perc = int(splt[5])
        self.default_constraint = Constraint(mv,mxv,perc)
        splt = splt[6::]
        splt = [(int(splt[i]),float(splt[i+1]),float(splt[i+2]),int(splt[i+3])) for i in range(0,len(splt),4)]
        self.constraints = {}
        for i in splt:
            self.constraints[i[0]] = Constraint(i[1],i[2],i[3])
class Breeder(ABC):
    _typename = ''
    def __init__(self,model):
        super().__init__() 
        if(model):
            self.set_model(self.model)
    @abstractmethod
    def next_generation(self, model,generation_id,population = None):
        pass
    def set_model(self,model):
        self.model = model
        self._set_model(model)
    @abstractmethod
    def _set_model(self, model):
        pass
    @abstractmethod
    def _select_pair(self,population):
        pass
    def _get_best_in(self,p):
        best= p[0]
        for i in p:
            if(i.evaluation>best.evaluation):
                best = i
        return best
    @abstractmethod
    def _do_rand_select(self,p,get_bad = False):
        pass
    def _get_conf_string(self):
        return self._typename
    def _load_conf(self,conf_string):
        return conf_string.split(",")
    @abstractmethod
    def _save(self,name):
        pass
    @abstractmethod
    def _load(self,name):
        pass
class Merger(ABC):
    _typename = ''
    def __init__(self,model):
        super().__init__()
        self.model =model
        if(self.model):
            self.set_model(self.model)
        pass
    @abstractmethod
    def merge(self,a,b):
        pass
    @abstractmethod
    def gen_gene(self,gene_size,default_constraint,constraints):
        pass
    def set_model(self,model):
        self.model = model
        self._set_model(model)
    @abstractmethod
    def _set_model(self, model):
        pass
    def _get_conf_string(self):
        return self._typename
    def _load_conf(self,conf_string):
        return conf_string.split(",")
    @abstractmethod
    def _save(self,name):
        pass
    @abstractmethod
    def _load(self,name):
        pass
### Breeders

#### Standard Breeder 
class StandardBreeder(Breeder):
    _typename = 'standard'
    def __init__(self,model,top_echelon = 0.2,b_rand=False):
        super().__init__(model)
        self.top_echelon = top_echelon
        self.b_rand = b_rand
    def _set_model(self, model):
        pass   
    def next_generation(self,generation_id,population = None):
        if(not hasattr(self,'model')):
            print("NO MODEL")
        npop = []
        if not population:
            population = self.model.population
        population.sort(key = lambda agent: agent.evaluation,reverse= True)
        rep = population[0:int(len(population)*self.top_echelon)]
        for j in range(len(population)-len(rep)):
            a,b=self._select_pair(rep)
            if(self.b_rand):
                npop2 = [i for i in population if i != a]
                b= random.choice(npop2)
            npop.append(self.model.merger.merge(a,b))
        return rep+npop,population[int(len(population)*self.top_echelon):]
    def _do_rand_select(self,p,get_bad = False):
        return random.choice(p)
    def _select_pair(self,population):
        npop = [i for i in population]
        a = self._do_rand_select(npop)
        npop.remove(a)
        b = self._do_rand_select(npop)
        return a,b

    def _get_conf_string(self):
        return super()._get_conf_string()+","+str(self.top_echelon)+","+str(self.b_rand)
    def _load_conf(self,conf_string):
        splt = conf_string.split(",")
        self.top_echelon = float(splt[1])
        self.b_rand = bool(splt[2])
        if(len(splt)>3):
            return splt[3::]
    def _save(self,name):
        pass
    def _load(self,name):
        pass
class WeightedSelectionBreeder(StandardBreeder):
    _typename = 'wsb'
    def __init__(self,model,top_echelon = 0.2,b_rand=False):
        super().__init__(model,top_echelon,b_rand)
    def _set_model(self,model):
        pass   
    def _do_rand_select(self,p,get_bad = False):
        weights = []
        total =0
        for i in p:
            total+=abs(i.evaluation)
        for i in range(len(p)):
            weights.append(p[i].evaluation/(max(total,1)))
            if get_bad:
                weights[-1] = 1-weights[-1]
        return random.choices([a for a in (p)],k=1,weights=weights)[0]    
    def _select_pair(self,pop):   
        npop = [i for i in pop]
        a = self._do_rand_select(npop)
        npop.remove(a)
        b = self._do_rand_select(npop)
        return a,b
class IslandBreeder(Breeder):
    _typename = 'island'
    def __init__(self,model,island_count = 5,top_echelon = 0.2,exchange_rate = 200,island_breeder = None,b_rand=False):
        super().__init__(model)
        self.b_rand = b_rand
        self.top_echelon = top_echelon
        self.island_count = 3
        self.exchange_rate = exchange_rate
        if(island_breeder == None):
            self.island_breeder = WeightedSelectionBreeder(model,self.top_echelon,self.b_rand)
        else:
            self.island_breeder = island_breeder
    def _set_model(self, model):
        self.island_breeder.set_model(model)
        if(not hasattr(self,'model')):
            print("NO MODEL IB")
    def _get_islands(self):
        island_size = len(self.model.population)//self.island_count
        islands = [[] for i in range(self.island_count)]
        
        for i in range(self.island_count):
            for j in range(island_size):
                islands[i].append(self.model.population[(i*island_size)+j])
            islands[i].sort(key = lambda agent: agent.evaluation,reverse= True)
        return island_size,islands
    def _set_islands(self,islands):
        self.island_count = len(islands)
        npop =[]
        for i in islands:
            npop+=i
        self.model.population = npop
    def next_generation(self,generation_id):
        island_size,islands = self._get_islands()
        npop = []
        removed = []
        if(generation_id % self.exchange_rate == 0 and generation_id!=0):
            ret= self._do_exchange(islands)
            if(ret):
                for i in range(self.island_count):
                    removed+=[islands[i].pop(-1) for j in range(len(ret[i]))]
                    islands[i]+=ret[i]
        for i in range(self.island_count):
            n,rem = self.island_breeder.next_generation(generation_id,islands[i]) 
            npop+=n
            removed+=rem
        
        return npop,removed
    def _do_exchange(self,islands):
        ret = []
        for i in range(len(islands)):
            ret.append([])
            for j in range(-1,2,1):
                if(j!=0):
                    ind = (i+j)%len(islands)
                    a = self.island_breeder._do_rand_select(islands[ind])
                    ret[i].append(a)
        return ret
    def _do_rand_select(self,pop,get_bad = False):
        return self.island_breeder._do_rand_select(pop)
    def _select_pair(self,pop):
        return self.island_breeder._select_pair(pop)
    def _get_conf_string(self):
        return super()._get_conf_string()+","+str(self.exchange_rate)+","+self.island_breeder._get_conf_string()
    def _load_conf(self,conf_string):
        rem = super()._load_conf(conf_string)
        self.exchange_rate = int(rem[1])
        self.island_breeder = breeder_from_string(rem[2])
        reconst = ""
        for i in range(2,len(rem)):
            reconst+=str(rem[i])
            if(i!=len(rem)-1):
                reconst+=","
        rem = self.island_breeder._load_conf(reconst)
        return rem
    def _save(self,name):
        pass
    def _load(self,name):
        pass
### Refreshing weigted selection island breeder
class RIB(IslandBreeder):
    _typename = 'rib'
    def __init__(self,model,island_count = 5,top_echelon = 0.2,exchange_rate = 200,refresh_rate = 500,island_breeder = None,b_rand = False):
        super().__init__(model,island_count,top_echelon,exchange_rate,island_breeder=island_breeder,b_rand=b_rand)
        self.refresh_rate = refresh_rate
        self.lock_down = 0
    def _set_model(self,model):
        self.island_breeder.set_model(model)
        #self.avgs  = self.get_new_averages()
        self.avgs = [0 for i in range(self.island_count)]
        
    def next_generation(self,generation_id):
        if(generation_id>5000):
            rr= self.refresh_rate*2
        else:
            rr=self.refresh_rate
        island_size,islands = self._get_islands()
        npop = []
        removed = []
        if(generation_id % rr == 0 and generation_id!=0):
            islands = self._do_refresh(islands)
        if(generation_id % self.exchange_rate == 0 and generation_id!=0):
            ret= self._do_exchange(islands)
            islands = ret
        for i in range(self.island_count):
            n,rem = self.island_breeder.next_generation(generation_id,islands[i]) 
            npop+=n
            removed+=rem
        
        return npop,removed    
    def get_new_averages(self):
        island_size, islands = self._get_islands()
        return [self.model.get_average_eval(islands[i]) for i in range(self.island_count)]
    def _do_refresh(self,islands):
        avgs = self.get_new_averages()
        min_developed = 0
        min_avg = abs(avgs[0]-self.avgs[0])
        for i in range(1,len(islands),1):
            if(abs(avgs[i]-self.avgs[i])<min_avg):
                min_avg = abs(avgs[i]-self.avgs[i])
                min_developed = i
        self.avgs = avgs
        islands[min_developed] = self.model._generate_random_pop(len(islands[min_developed]))
        self.lock_down= min_developed
        return islands
    def _do_exchange(self,islands):
        
        for i in range(len(islands)):
            islands[i].sort(key=lambda x: x.evaluation, reverse = True)
        for i in range(len(islands)):
            if(self.lock_down==i):
                pass
                
            for j in range(-1,2,1):
                if(j!=0):
                    ind = (i+j)%len(islands)
                    a = self.island_breeder._do_rand_select(islands[ind])
                    islands[i].pop(-1)
                    islands[i].append(a)
        return islands
    def _get_conf_string(self):
        return super()._get_conf_string()+","+str(self.refresh_rate)
    def _load_conf(self,conf_string):
        rem = super()._load_conf(conf_string)
        self.refresh_rate = int(rem[0])
        if(len(rem)>1):
            return rem[1::]
#### Standard Merger 
class StandardMerger(Merger):
    _typename = 'standard'
    def __init__(self,model,crossover = 0.5,mutation_chance = 0.5,mutation_rate =1):
        super().__init__(model)
        self.crossover=crossover
        self.mutation_chance = mutation_chance
        self.mutation_rate = mutation_rate
    def _set_model(self, model):
        pass
    def merge(self,a,b):
        ng = []
        if(a.gene_size<b.gene_size):
            b,a = a,b
        for i in range(a.gene_size):
            if(i<b.gene_size):
                r = random.random()
                if(r<self.crossover):
                    ng.append(a.dna[i])
                else:
                    ng.append(b.dna[i])
            else:
                ng.append(a.dna[i]) 
            constraint = self.model.default_constraint
            if(i in self.model.constraints):
                constraint = self.model.constraints[i]
            if(random.random()<self.mutation_chance):
                delta = constraint.random(ranged=True)
                ng[-1] += delta
                ng[-1] = constraint.apply(ng[-1])

        ret = self.model._generate_agent(ng,a,b)
        return ret
    def gen_gene(self,gene_size,default_constraint=Constraint(0,2,100),constraints={}):
        ret = []
        for i in range(gene_size):
            constraint = default_constraint
            if(i in constraints):
                constraint = constraints[i]
            ret.append(constraint.random(ranged=True))
        return ret
    def _get_conf_string(self):
        return super()._get_conf_string()+","+str(self.crossover)+","+str(self.mutation_chance)+","+str(self.mutation_rate)
    def _load_conf(self,conf_string):
        splt = conf_string.split(',')
        self.crossover = float(splt[1])
        self.mutation_chance = float(splt[2])
        self.mutation_rate = float(splt[3])
        if(len(splt)>4):
            return splt[4::]
    def _save(self,name):
        pass
    def _load(self,name):
        pass
class MomentumMerger(Merger):
    _typename = 'm_merger'
    alpha = 0.5 # reverse chance
    beta = 0.7 # increment momentum chance
    omega = 0.1 # hold mementum chance (chance to decrement momentum is 1-(beta+omega))
    def __init__(self,model,crossover = 0.5,mutation_chance = 0.5,mutation_rate =8,max_mutations = 0,decay_every = 1,alpha=None,beta =None,omega= None):
        super().__init__(model)
        self.max_mutations = max_mutations
        self.crossover=crossover
        self.mutation_chance = mutation_chance
        self.mutation_rate = mutation_rate
        self.momentums = {}
        self.decay_every =decay_every
        if(alpha):
            self.alpha=alpha
        if(beta):
            self.beta = beta
        if(omega):
            self.omega = omega
    def _set_model(self, model):
        self.model.on_agents_removed.add(self.remove_killed)
        self.model.on_time_step.add(self.decay_momentums)
    def decay_momentums(self,genid):
        if(self.model.tdelta%self.decay_every!=0):
            return
        for i in self.momentums:
            d = self.momentums[i]
            for j in range(len(d)):
                d[j] = math.copysign(max(1,abs(d[j])-1),d[j])
                if(abs(d[j])==1 and random.random()<self.alpha):
                        d[j]*=-random.randint(1,self.model.get_constraint(i).max_accuracy//2)
    def remove_killed(self,killed):
        for i in killed:
            if(i.id in self.momentums):
                del self.momentums[i.id]
    def merge(self,a,b):
        ng = []
        if(a.gene_size<b.gene_size):
            b,a = a,b
        for i in range(a.gene_size):
            if(i<b.gene_size):
                r = random.random()
                if((abs(self._get_momentum(a,i))>=abs(self._get_momentum(b,i)) and isinstance(self.crossover,str) and self.crossover == 'dynamic') or (isinstance(self.crossover,float) and r<self.crossover)):
                    ng.append(a.dna[i])
                else:
                    ng.append(b.dna[i])
            else:
                ng.append(a.dna[i]) 
        ret = self.model._generate_agent(ng,a,b)
        self.create_momentum(ret,a,b)
        self._do_mutate(ret)
        return ret
    def create_momentum(self,child,agent_a=None,agent_b=None):
        momentum = []
        if(agent_a and agent_b):
            for i in range(len(child.dna)):
                adist= abs(child.dna[i]-agent_a.dna[i])
                bdist= abs(child.dna[i]-agent_b.dna[i])
                if(bdist>adist):
                    momentum.append(self._get_momentum(agent_a,i))
                else:
                    momentum.append(self._get_momentum(agent_b,i))
                #momentum.append(max(self._get_momentum(agent_a,i),self._get_momentum(agent_b,i)))
                
        else:
            momentum = [math.copysign(1,1-(random.random()*2)) for i in range(len(child.dna))]
        self.momentums[child.id]=momentum
    def _do_mutate(self,ret):
        ng = ret.dna
        muts = 0     
        while random.random() < self.mutation_chance and (self.max_mutations==0 or muts <self.max_mutations):
            i =self._selectMutate(ret)
            mome = self._get_momentum(ret,i)
            constraint = self.model.default_constraint
            if(i in self.model.constraints):
                constraint = self.model.constraints[i]
            #delta = round(random.random()/(10**abs(mome)),abs(int(mome))+1)*self.mutation_rate
            delta = math.copysign((constraint.random(ranged=True)*self.mutation_rate)/(10**abs(mome)),mome)
            ng[i] += math.copysign(delta,mome)
            ng[i] = constraint.apply(ng[i])
            self._update_momentums(ret,i,delta)
            muts+=1
        ret.dna=ng
    def _get_momentum(self,agent,i):
        if(agent.id in self.momentums):
            return self.momentums[agent.id][i]
        else:
            self.create_momentum(agent)
            return self._get_momentum(agent,i)
    def _update_momentums(self,agent,i,value):
        constraint = self.model.get_constraint(i)
        if(random.random()>self.beta):
            self.momentums[agent.id][i] = math.copysign(min(abs(self.momentums[agent.id][i])+2,constraint.max_accuracy),value)
        elif(random.random()>self.omega):
            self.momentums[agent.id][i] = math.copysign(min(abs(self.momentums[agent.id][i])+1,constraint.max_accuracy),value)
        
    def _selectMutate(self,agent):
        self._get_momentum(agent,0)
        momentum = self.momentums[agent.id]
        total = 0
        for i in momentum:
            total+=math.log10(abs(i)+1)
        total=max(total,1)
        weights = [math.log10(abs(i)+1)/total for i in momentum]
        return random.choices([(i,j) for i,j in enumerate(agent.dna)],weights=weights,k=1)[0][0]

    def gen_gene(self,gene_size,default_constraint=Constraint(0,1,100),constraints={}):
        ret = []
        for i in range(gene_size):
            constraint = default_constraint
            if(i in constraints):
                constraint = constraints[i]
            ret.append(constraint.random(ranged=True))
        return ret 
    def _get_conf_string(self):
        return super()._get_conf_string()+","+str(self.decay_every)+","+str(self.max_mutations)+","+str(self.alpha)+","+str(self.beta)+","+str(self.omega)
    def _load_conf(self,conf_string):
        rem = super()._load_conf(conf_string)
        self.decay_every = int(rem[1])
        self.max_mutations = int(rem[2])
        self.alpha = float(rem[3])
        self.beta = float(rem[4])
        self.omega = float(rem[5])
        if(len(rem)>6):
            return rem[6::]
    def _save(self,name):
        with open("%s_momentum.bin" % name,'wb') as f:
            gs = self.model.gene_size
            c = len(self.momentums)
            f.write(struct.pack('>l',c))
            f.write(struct.pack('>l',gs))
            for i in self.momentums:
                f.write(struct.pack('>l',i))
                f.write(struct.pack('>'+('f'*gs),*self.momentums[i]))
    def _load(self,name):
        with open("%s_momentum.bin" % name,'rb') as f:
            c=  struct.unpack('>l',f.read(4))[0]
            gs=  struct.unpack('>l',f.read(4))[0]
            for i in range(c):
                ag = struct.unpack('>l',f.read(4))[0]
                m = [i for i in struct.unpack('>'+('f'*gs),f.read(4*gs))]
                self.momentums[ag]=m

    
#### Dictionary
_breeders = {
    'standard' : StandardBreeder,
    'wsb' : WeightedSelectionBreeder,
    'island' : IslandBreeder,
    'rib' : RIB
}
_mergers = {
    'standard' : StandardMerger,
    'm_merger' : MomentumMerger
}