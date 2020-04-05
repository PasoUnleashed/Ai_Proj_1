from multiprocessing import Pool,freeze_support,Manager
def f(x,q):
    q.put(x)
if __name__=='__main__':
    freeze_support()
    m = Manager()
    q = m.Queue()
    with Pool(5) as p:
        p.starmap(f,[(i,q) for i in range(10)])
    while(q.qsize()<10):
        print(q.qsize())
    [print(q.get()) for i in range(10)]