from benchmark.configuration_space_tmp import Classification_Configuration_Space
import optuna
import matplotlib.pyplot as plt
from scipy.stats import uniform,loguniform
from hyperopt import hp
import ConfigSpace as CS
import optuna
from optuna.samplers import TPESampler,RandomSampler


# Define a simple 2-dimensional objective function whose minimum value is -1 when (x, y) = (0, -1).
def objective(trial,values):
    x = trial.suggest_float("x", 2**-10, 2**10, log=True)
    values.append(x)
    return x**2


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
    study.optimize(lambda x : objective(x,values), n_trials=100) 
    plt.hist(values,label='optuna' +str(seed))

if __name__ == "__main__":
    # Let us minimize the objective function above.
    print("Running 10 trials...")
    
    for i in [1,2,3,4,5]:
        create_plot_optuna(i)
    plt.legend()
    plt.show()
