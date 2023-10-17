import optuna
import matplotlib.pyplot as plt
from scipy.stats import uniform,loguniform
from hyperopt import hp
import hyperopt
import ConfigSpace as CS
import optuna
from optuna.samplers import TPESampler,RandomSampler
from mango import MetaTuner
import numpy as np

# Define a simple 2-dimensional objective function whose minimum value is -1 when (x, y) = (0, -1).
def objective_optuna(trial,values):
    x = trial.suggest_float("x", 2**-10, 2**10, log=True)
    values.append(x)
    return x**2

def objective_mango(args_list):
    results = []
    for hyper_par in args_list:
        x = hyper_par['x']
        results.append(x)
        
    return results

def create_mango_optimizer(seed):
    configspace = dict({'x':  loguniform(2**-10, 2**10)})
    res=MetaTuner([configspace], [objective_mango],n_init=100,n_iter = 0, seed= seed).run()
    a = seed/10
    plt.hist(res['objective_values'],label='mango' +str(seed),fc=(0, 0, 1*a, a))


"""uniform(-100, 100)
hp.uniform(name, lower_bound, upper_bound)
CS.UniformFloatHyperparameter(name, lower=lower_bound, upper=upper_bound, log=True)"""

def create_optimizer_optuna(seed):
    sampler = RandomSampler(seed=seed)
    #Don't prune
    pruner= optuna.pruners.NopPruner()
    study = optuna.create_study(direction='minimize',sampler=sampler,pruner=pruner)
    return study


def create_plot_optuna(seed):
    study = create_optimizer_optuna(seed)
    values = []
    study.optimize(lambda x : objective_optuna(x,values), n_trials=100) 
    a = seed/10
    plt.hist(values,label='optuna' +str(seed),fc=(1*a, 1*a, 1*a, a))

def return_x(config):
    print(config)
    return config.get_dictionary()['x']

def hyperopt_objective(config):
    return config['x']

def create_hyperopt(seed):
    config_space = {'x' : hp.loguniform('x',np.log(2**-10),np.log(2**10))}
    trials = hyperopt.Trials()
    hyperopt.fmin(hyperopt_objective, config_space,algo=hyperopt.tpe.suggest,trials=trials, max_evals=100, rstate=np.random.default_rng(seed),show_progressbar=True)
    values = np.array([trial['loss'] for trial in trials.results])
    a = seed/10
    plt.hist(values,label='hyperopt' +str(seed),fc=(1*a, 1*a, 0, a))


def create_smac(seed): 
    cs = CS.ConfigurationSpace(seed=seed)
    
    cs.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=2**-10, upper=2**10, log=True))
    configurations = cs.sample_configuration(100)
    #print(configurations)
    values = list(map(return_x,configurations))
    a = seed/10
    plt.hist(values,label='smac' +str(seed),fc=(0, 1*a, 1*a, a))

if __name__ == "__main__":
    # Let us minimize the objective function above.
    print("Running 10 trials...")
    
    for i in range(1,2):
        create_plot_optuna(i)
        create_mango_optimizer(i)
        create_smac(i)
        create_hyperopt(i)
    plt.legend()
    plt.show()
