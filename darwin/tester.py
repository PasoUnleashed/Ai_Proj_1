import darwin
import math
import random
### Model Test 1:

def shwefels(dna):
    total=0
    for i in range(len(dna)):
       total += dna[i]*math.sin(math.sqrt(abs(dna[i])))
    return (418.9829*len(dna))-total        
def sphere(dna):
    total = 0
    for i in dna:
        total+=i**2
    return total
def rosenbrock(dna):
    total = 0
    for i in range(len(dna)-1):
        x = 100
        x*=(dna[i+1]-(dna[i]**2))**2
        x+=(dna[i]-1)**2
        total+=x
    return total
def ackley(dna):
    n = len(dna)
    sxi2 = 0
    cosm = 0
    for i in dna:
        sxi2+=i**2
        cosm+=math.cos((2*math.pi)*i)
    sxi2 = 20*math.exp(-0.2*math.sqrt((1/n)*sxi2))
    cosm = math.exp((1/n)*cosm)
    return 20 + math.e - sxi2 - cosm
def rastrigin(dna):
    sm = 0
    for i in dna:
        sm+=(i**2)-(10*math.cos(math.pi*2*i))
    return (10*len(dna))+sm
def griewank(dna):
    sm =0
    prod = 1
    for i in range(len(dna)):
        sm+=dna[i]**2
        prod *=math.cos(dna[i]/math.sqrt((i+1)))
    sm*=1/4000
    return (1+sm)-prod 

### Benchmark Results ###


#             |Converges? |
# Function    | RIB | WSB |
# Rosenbrook  |  Y  |  Y  |
# Sphere      |  Y  |  Y  |
# Rastrigin   |  Y  |  Y  |
# Griewank    | Sometimes Y  |  Sometimes Y  |
# Shwefel's   |  Y  |  Mostly Y  |

# General setup:

'''
breeder = darwin.RIB(model = None,island_count =5,crossover = 0.7,top_echelon = 0.2,exchange_rate=250,refresh_rate=250)
breeder = darwin.WeightedSelectionBreeder(model = None,crossover = 0.5,top_echelon = 0.5)
merger = darwin.MomentumMerger(model = None,crossover = 0.7,mutation_chance = 0.6,mutation_rate =1,max_mutations =1)
model = darwin.StaticEvaluationModel(pop_size=300,breeder =breeder,merger = merger,gene_size = gene_size,default_constraint=constraint,constraints={})

'''

# Griwank setup:
'''
breeder = darwin.RIB(model = None,island_count =3,top_echelon = 0.05,exchange_rate=125,refresh_rate=500)
#breeder = darwin.WeightedSelectionBreeder(model = None,crossover = 0.5,top_echelon = 0.5)
merger = darwin.MomentumMerger(model = None,crossover = 0.7,mutation_chance = 0.8,mutation_rate =10,max_mutations =0)
model = darwin.StaticEvaluationModel(pop_size=120,breeder =breeder,merger = merger,gene_size = gene_size,default_constraint=constraint,constraints={})
'''

# shwefel's setup:
'''
 breeder = darwin.RIB(model = None,island_count =3,top_echelon = 0.25,exchange_rate=125,refresh_rate=500)
#breeder = darwin.WeightedSelectionBreeder(model = None,crossover = 0.5,top_echelon = 0.25)
#breeder = darwin.StandardBreeder(model = None,crossover=0.5,top_echelon=0.25)
merger = darwin.MomentumMerger(model = None,crossover = 0.7,mutation_chance = 0.8,mutation_rate =12,max_mutations =0)
model = darwin.StaticEvaluationModel(pop_size=120,breeder =breeder,merger = merger,gene_size = gene_size,default_constraint=constraint,constraints={})
'''
#####
def model1_benchamrk(n,function,constraint,max_steps = 35000):
    testac= 20
    gene_size = n
    #breeder = darwin.RIB(model = None,island_count =3,top_echelon = 0.334,exchange_rate=500,refresh_rate=500,b_rand=True)
    breeder = darwin.WeightedSelectionBreeder(model = None,top_echelon = 0.3,b_rand=True)
    merger = darwin.MomentumMerger(model = None,crossover = 'dynamic',mutation_chance = 0.5,mutation_rate =20,max_mutations =0,decay_every=1)
    model = darwin.StaticEvaluationModel(pop_size=300,breeder =breeder,merger = merger,gene_size = gene_size,default_constraint=constraint,constraints={})
    error = float('inf')
    data = []
    while(round(error,testac)>1e-5 and max_steps>0):
        for i in model.population:
            evalu = abs(function(i.dna))
            evalu1 = 1-evalu
            i.evaluate(evalu1)
            error=min(evalu,error)
        data.append(error)
        avg = model.get_average_eval(model.population)
        model.TimeStep()
        print(("Generation: #%d"% model.tdelta).ljust(20),"|",("Error: %s" % str(round(error,testac))).ljust(20),"|",("Avg. Accuracy: %s" % str(round(avg,8))).rjust(15))
        max_steps-=1
    bst = model.get_best()
    print([round(i,testac) for i in bst.dna])
    print([round(i,2) for i in bst.dna])
    return error,data,model.get_best(),model.tdelta
def model1_test_save(n,function,constraint,max_steps = 35000):
    testac= 20
    gene_size = n
    breeder = darwin.RIB(model = None,island_count =3,top_echelon = 0.334,exchange_rate=500,refresh_rate=500,b_rand=True)
    #breeder = darwin.WeightedSelectionBreeder(model = None,top_echelon = 0.3,b_rand=True)
    merger = darwin.MomentumMerger(model = None,crossover = 'dynamic',mutation_chance = 0.5,mutation_rate =6,max_mutations =0,decay_every=1)
    model = darwin.StaticEvaluationModel(pop_size=30,breeder =breeder,merger = merger,gene_size = gene_size,default_constraint=constraint,constraints={})
    error = float('inf')
    data = []
    while(round(error,testac)>1e-5 and max_steps>0):
        if(max_steps%100==0):
            model.save('test')
            model.load('test')
        for i in model.population:
            evalu = abs(function(i.dna))
            evalu1 = 1-evalu
            i.evaluate(evalu1)
            error=min(evalu,error)
        data.append(error)
        avg = model.get_average_eval(model.population)
        model.TimeStep()
        print(("Generation: #%d"% model.tdelta).ljust(20),"|",("Error: %s" % str(round(error,testac))).ljust(20),"|",("Avg. Accuracy: %s" % str(round(avg,8))).rjust(15))
        max_steps-=1
    bst = model.get_best()
    print([round(i,testac) for i in bst.dna])
    print([round(i,2) for i in bst.dna])
    return error,data,model.get_best(),model.tdelta
def benchmark_and_save(epochs, generations,dimension,function,constraint,constraints):
    data = [0 for i in range(generations)]
    error_total = 0
    gens_total =0 
    passed = 0
    for i in range(epochs):
        error,ret_data,bst,gens = model1_benchamrk(dimension,function,constraint,generations)
        gens_total+=gens
        for j in range(generations):
            if(j<len(ret_data)):
                data[j]+=ret_data[j]
            else:
                data[j]+=error
        error_total+=error
        if(gens<generations):
            passed+=1
    error_total/=epochs
    gens_total/=epochs
    data = [i/epochs for i in data]
    print(error_total,gens_total,"%d/%d" %(passed,epochs))

#model1_benchamrk(2,rosenbrock,darwin.Constraint(-2.048,2.048,10)) 
model1_benchamrk(1000,sphere,darwin.Constraint(-5.12,5.12,9))
#model1_benchamrk(20,rastrigin,darwin.Constraint(-5.12,5.12,20))
#model1_benchamrk(10,shwefels,darwin.Constraint(-500,500,8))
#model1_benchamrk(10,ackley,darwin.Constraint(-30,30,8))
#model1_benchamrk(10,griewank,darwin.Constraint(-600,600,16))
#benchmark_and_save(100,5000,2,griewank,darwin.Constraint(-600,600,16),{})

#model1_test_save(2,rosenbrock,darwin.Constraint(-2.048,2.048,10)) 