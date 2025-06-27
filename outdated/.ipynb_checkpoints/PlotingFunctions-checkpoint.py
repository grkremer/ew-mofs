import numpy as np
import matplotlib.pyplot  as plt
from pymoo.indicators.hv import HV

def plotSingleResult(res):
    n_evals = np.array([e.evaluator.n_eval for e in res.history])
    opt = np.array([e.opt[0].F for e in res.history])
        
    X_res, F_res = res.opt.get("X", "F")
        
    hist = res.history
    max = 100
    ref_point = np.array([max, -0.90])
    ind = HV(ref_point=ref_point)
        
    n_evals = []             # corresponding number of function evaluations\
    hist_F = []              # the objective space values in each generation
    hist_cv = []             # constraint violation in each generation
    hist_cv_avg = []         # average constraint violation in the whole population
        
    for algo in hist:
        # store the number of function evaluations
        n_evals.append(algo.evaluator.n_eval)
        
        # retrieve the optimum from the algorithm
        opt = algo.opt
        
        # store the least contraint violation and the average in each population
        hist_cv.append(opt.get("CV").min())
        hist_cv_avg.append(algo.pop.get("CV").mean())
        
        # filter out only the feasible and append and objective space values
        feas = np.where(opt.get("feasible"))[0]
        hist_F.append(opt.get("F")[feas])
    
    metric = HV(ref_point= np.array([max, 0]),
                             norm_ref_point=False)
    
    hv = [metric.do(_F)/max for _F in hist_F]
    
        #plt.figure(figsize=(7, 5))
    plt.plot(n_evals, hv,  color='black', lw=0.7, label="Avg. CV of Pop")
    plt.scatter(n_evals, hv,  facecolor="none", edgecolor='black', marker="p")
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    return hv[-1]

def plot_convergence(res, color):
    n_evals = np.array([e.evaluator.n_eval for e in res.history])
    opt = np.array([e.opt[0].F for e in res.history])
    
    X_res, F_res = res.opt.get("X", "F")
    
    hist = res.history
    max = 50
    ref_point = np.array([max, -0.90])
    ind = HV(ref_point=ref_point)
    
    n_evals = []             # corresponding number of function evaluations\
    hist_F = []              # the objective space values in each generation
    hist_cv = []             # constraint violation in each generation
    hist_cv_avg = []         # average constraint violation in the whole population
    
    for algo in hist:
        # store the number of function evaluations
        n_evals.append(algo.evaluator.n_eval)
    
        # retrieve the optimum from the algorithm
        opt = algo.opt
    
        # store the least contraint violation and the average in each population
        hist_cv.append(opt.get("CV").min())
        hist_cv_avg.append(algo.pop.get("CV").mean())
    
        # filter out only the feasible and append and objective space values
        feas = np.where(opt.get("feasible"))[0]
        hist_F.append(opt.get("F")[feas])

    metric = HV(ref_point= ref_point,
                         norm_ref_point=False)

    hv = [metric.do(_F)/5 for _F in hist_F]

    plt.plot(n_evals, hv,  color='black', lw=0.7, label="Avg. CV of Pop")
    plt.scatter(n_evals, hv,  facecolor="none", edgecolor=color, marker="p", linewidths=1)
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    for i in range(20):
        
        plt.axvline(x=200*i, color='r', linestyle='--', linewidth=0.7)
    #plt.xscale(value = 'log')
    #plt.show()
    return ind(F_res)/5


def plotComṕarisonConvergence(result_stock, result_modified):
    plt.figure(figsize=(16,6))
    media_stock = []
    for res in result_stock:
        media_stock.append(plot_convergence(res, 'blue'))
    
    #plt.figure(figsize=(20,6))
    print(np.array(media_stock).sum()/len(result_stock))
    
    media_modified = []
    for res in result_modified:
        media_modified.append(plot_convergence(res, 'red'))
        
    print(np.array(media_modified).sum()/len(result_stock))
    