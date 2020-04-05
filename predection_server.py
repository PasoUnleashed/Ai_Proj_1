
import leaderboard
import os
import gameplayer
import multiprocessing as mp
class Worker:
    def __init__(self,id_,server):
        m = mp.Manager()
        self.outgoing = m.Queue()
        self.incoming = m.Queue()
        self.process = mp.Process(target = gameplayer.Client,args = (id_,self.incoming,self.outgoing))
        self.server = server
        self.id = id_
        self.status = 'stopped'
    def wait_message(self):
        while(self.incoming.empty()):
            pass
        return self.incoming.get()
    def start(self):
        if(self.status!='stopped'):
            raise Exception("Attempted to started an already running worker")
        self.process.start()
        m = self.wait_message()
        if(m[0] == 'ready'):
            self.status = 'idle'
        
    def play_match(self,match):
        p1,p2 = match
        self.match = match
        self.outgoing.put(('play',p1.id,p2.id))
        x =self.wait_message()
        if(x[0] == 'started'):
            self.status = 'playing'
        else:
            self.status = 'N/A'
    def process_queue(self):
        if(not self.incoming.empty()):
            command = self.incoming.get()
            flag = command[0]
            if(flag == 'result'):
                self.server.match_complete(self,self.match,command[1])
                self.status='idle' 

class Server:
    on_match_complete = set()
    def __init__(self,worker_count,create_model_function,name):
        self.models = {}
        self.name = name
        self.gene_size = leaderboard.cal_model_wlen(create_model_function())
        self.workers = []
        self.worker_count =worker_count
        self.create_model_function=create_model_function
    def match_complete(self,worker,match,result):
        for i in self.on_match_complete:
            i(match,result)
    def start(self):
        self.spawn_workers()
        print("Prediction Server Started")
    def spawn_workers(self):
        leaderboard.printProgressBar(0,self.worker_count,prefix='Starting Workers..'.ljust(17),suffix="%d/%d"%(0,self.worker_count))
        for i in range(self.worker_count):
            self.workers.append( Worker(i,self))
            self.workers[i].start()
            self.workers[i].outgoing.put(('gs',self.gene_size))
            self.workers[i].outgoing.put(('lb',self.name))
            leaderboard.printProgressBar(i+1,self.worker_count,prefix='Starting Workers..'.ljust(17),suffix="%d/%d"%(i+1,self.worker_count))
    def play_matches(self,matches):
        mq = list(matches)
        while(len(mq)>0):
            for i in self.workers:
                if(len(mq)==0):
                    break
                if(i.status == 'idle'):
                    i.play_match(mq.pop())
                    leaderboard.printProgressBar(len(matches)-len(mq),len(matches),prefix='Playing...'.ljust(17),suffix="%d/%d"%(len(matches)-len(mq),len(matches)))
                else:
                    i.process_queue()
        self.wait_all()
    def wait_all(self):
        x=0
        while(True):

            x+=1
            active =False
            for i in self.workers:
                i.process_queue()
                if(i.status!='idle'):
                    active=True
            if(not active):
                break
if __name__ == '__main__':
    import keras