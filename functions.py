### PYMOO
from pymoo.core.problem import Problem, ElementwiseProblem, StarmapParallelization
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.indicators.hv import HV
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination import get_termination
from pymoo.core.crossover import Crossover
from pymoo.util.misc import crossover_mask

#multiprocessamento e outros
from tqdm.notebook import trange, tqdm
import multiprocessing, requests, sys, time, itertools, dill, random, os, pickle, copy

#Pandas, SKLearn e etc.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics, svm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skrebate import ReliefF

from ucimlrepo import fetch_ucirepo 

model = make_pipeline(
        StandardScaler(),
        DecisionTreeClassifier(max_depth=6))
CLASSIFIER = model
MIN_FEATURES = 1
MAX_FEATURES = 100
N_PROCESS = 100

### Sampling
class BinaryRandomSampling(Sampling):
    def __init__(self, **kwargs):
        self.max_features = kwargs.pop('max_features', 100)
        super().__init__(**kwargs)
    def _do(self, problem, n_samples, **kwargs):
        val = np.random.random((n_samples, problem.n_var))
        val = checkEmpty(np.array((val < 0.5).astype(bool)), 100)
        return val
        
class BinaryDistributedWeightsSampling(Sampling):
    def __init__(self, **kwargs):
        self.sampling_weights = kwargs.pop('sampling_weights', None)
        self.seed = kwargs.pop('seed', 42)
        self.max_features = kwargs.pop('max_features', 100)
        super().__init__(**kwargs)
        
    def _do(self, problem, n_samples, **kwargs):
        random.seed(self.seed)
        population = []
        for i in range(n_samples):
            trues = np.random.randint(1, min(self.max_features, problem.n_var) + 1)
            individual = np.full(problem.n_var, False)
            individual[random.choices(range(problem.n_var), weights = self.sampling_weights, k=trues)] = True
            population.append(individual)
        population = np.array(population)
        return population

class SparseEASampling(Sampling):
    def __init__(self, **kwargs):
        self.sc = kwargs.pop('sc', None)
        self.seed = kwargs.pop('seed', 42)
        self.max_features = kwargs.pop('max_features', 100)
        super().__init__(**kwargs)
        
    def _do(self, problem, n_samples, **kwargs):
        random.seed(self.seed)
        population = []
        for i in range(n_samples):
            trues = np.random.randint(1, min(self.max_features, problem.n_var)  + 1)
            individual = np.full(len(self.sc), False)
            
            for j in range(trues):
                candidates = random.choices(range(len(self.sc)), k = 2)
                if self.sc[candidates[0]] > self.sc[candidates[1]]:
                    individual[candidates[0]] = True
                else:
                    individual[candidates[1]] = True

            population.append(individual)
        population = np.array(population)
        row_sums = population.sum(axis=1)
        # Mostra a matriz ordenada
        return population

class ReliefFSampling(Sampling):
    def __init__(self, **kwargs):
        self.sc = kwargs.pop('sc', None)
        self.seed = kwargs.pop('seed', 42)
        self.max_features = kwargs.pop('max_features', 100)
        super().__init__(**kwargs)
        
    def tournament_selection(self, num, sc):
        selected_indices = []
        for _ in range(num):
            # Escolhe dois índices aleatórios
            idx1, idx2 = np.random.choice(len(sc), size=2, replace=False)
            # Compara as pontuações e seleciona o índice com a maior pontuação
            if sc[idx1] > sc[idx2]:
                selected_indices.append(idx1)
            else:
                selected_indices.append(idx2)
        # Remove duplicatas e retorna os índices únicos
        return list(set(selected_indices))

    def _do(self, problem, n_samples, **kwargs):
        population = np.full((n_samples, len(self.sc)), False)  # Inicializa a população com zeros
        for i in range(n_samples):
            num = np.random.randint(1, min(self.max_features, problem.n_var)  + 1)
            selected_indices = self.tournament_selection(num, self.sc)
            population[i, selected_indices] = True
        return population

### Mutation
def checkEmpty(population, max_features):
    max_restriction = True
    new_pop = []
    for individual in population:
        if individual.sum() == 0:
            individual[np.random.randint(0,(len(individual)))] = True
        if individual.sum() > max_features:
            true_indices = np.array(list(range(len(individual))))[individual]
            individual = np.full(len(individual), False)
            individual[random.choices(true_indices, k = max_features)] = True
        new_pop.append(individual)
    return np.array(new_pop)

class BitflipMutationLimitedBalanced(Mutation):
    def __init__(self, **kwargs):
        self.weights = kwargs.pop('weights', None)
        self.max_features = kwargs.pop('max_features', 100)
        super().__init__(**kwargs)
        
    def _do(self, problem, X, **kwargs):
        prob_var = self.get_prob_var(problem, size=(len(X), 1))
        full = np.random.rand(*X.shape) < prob_var  # Evita uma criação redundante de matriz e faz o flip diretamente

        # Itera diretamente em X e full para alterar os valores
        for individual, vector in zip(X, full):
            num_changes = min(self.max_features, np.sum(vector))  # Limita a quantidade de alterações pelo MAX_FEATURES
            true_indices = np.where(individual)[0]
            false_indices = np.where(~individual)[0]
        
            # Seleciona índices aleatórios para trocar, dividindo o número de mudanças igualmente entre 'True' e 'False'
            flip_choices = np.random.rand(num_changes) < 0.5
            true_flips = np.sum(flip_choices)
            false_flips = num_changes - true_flips

            if self.weights is not None:
                weights = self.weights[false_indices]
            else:
                weights = None
                
            if true_flips > 0 and len(true_indices) > 0:
                individual[np.random.choice(true_indices, min(true_flips, len(true_indices)), replace=False)] = False
            if false_flips > 0 and len(false_indices) > 0:
                individual[random.choices(false_indices, k = min(false_flips, len(false_indices)), weights=weights)] = True
        X = checkEmpty(X, self.max_features)
        return X


class SparseEAMutation(Mutation):
    def __init__(self, **kwargs):
        self.weights = kwargs.pop('weights', None)
        self.max_features = kwargs.pop('max_features', 100)
        super().__init__(**kwargs)
        
    def _do(self, problem, X, **kwargs):
        # Itera diretamente em X e full para alterar os valores
        for individual in X:
            true_indices = np.where(individual)[0]
            false_indices = np.where(~individual)[0]
        
            if np.random.rand() < 0.5:
                # Seleciona dois índices de variáveis não nulas
                if len(true_indices) >= 2:
                    m, n = np.random.choice(true_indices, size=2, replace=False)
                    # Aplica a lógica baseada no Score
                    if self.weights[m] < self.weights[n]:
                        individual[m] = 0
                    else:
                        individual[n] = 0
            else:
                # Seleciona dois índices de variáveis nulas
                if len(false_indices) >= 2:
                    m, n = np.random.choice(false_indices, size=2, replace=False)
                    # Aplica a lógica baseada no Score
                    if self.weights[m] > self.weights[n]:
                        individual[m] = 1
                    else:
                        individual[n] = 1

        for individual in X:
            if individual.sum() == 0:
                individual[np.random.choice(range(len(individual)))] = True
            while individual.sum() > MAX_FEATURES:
                individual = individual[np.random.choice(np.where(individual == True)[0], MAX_FEATURES)]
        X = checkEmpty(X, self.max_features)
        return X

class ReliefFMutation(Mutation):
    def __init__(self, **kwargs):
        self.sc = kwargs.pop('sc', None)
        self.max_features = kwargs.pop('max_features', 100)
        super().__init__(**kwargs) # Scores das features do ReliefF

    def _do(self, problem, X, **kwargs):
        X_ = []
        for individual in X:
            selected_indices = np.where(individual == True)[0]
            unselected_indices = np.where(individual == False)[0]
            
            # 1. Decidir se remove ou adiciona uma feature
            if np.random.rand() < 0.5:
                # Mutação Tipo 1: Remover uma feature selecionada (menor score)
                if len(selected_indices) > 1:
                    # Torneio binário invertido (prioriza features com scores BAIXOS)
                    scores = -self.sc[np.array(selected_indices)]  # Inverte os scores
                    winner = self._binary_tournament(selected_indices, scores)
                    individual[winner] = False  # Remove a feature
            elif len(selected_indices) < MAX_FEATURES:
                # Mutação Tipo 2: Adicionar uma feature não selecionada (maior score)
                unselected_indices = np.where(individual == False)[0]
                if len(unselected_indices) > 0:
                    # Torneio binário normal (prioriza features com scores ALTOS)
                    scores = self.sc[unselected_indices]
                    winner = self._binary_tournament(unselected_indices, scores)
                    individual[winner] = True  # Adiciona a feature
            
            X_.append(individual)
        X_ = checkEmpty(X_, self.max_features)
        return np.array(X_)

    def _binary_tournament(self, candidates, scores):
        # Escolhe 2 candidatos aleatoriamente e seleciona o com maior score
        idx = np.random.choice(len(candidates), 2, replace= True if len(candidates) == 1 else False) #Quanto tem só 1 individuo, o replace precisa ser True
        winner = candidates[idx[np.argmax(scores[idx])]]
        return winner

### Crossover
class SparseEACrossover(Crossover):

    def __init__(self, **kwargs):
        self.sc = kwargs.pop('sc', None)
        super().__init__(2, 2, **kwargs)

    def _do(self, _, X, **kwargs):
        _, n_matings, n_var = X.shape
        M = np.random.random((n_matings, n_var)) < 0.5
        X_ = list(np.copy(X[0]))
        X_.extend(list(np.copy(X[1])))
        X_.extend(X_)
        O = []
        while X_:
            # Seleciona dois pais aleatoriamente
            p_idx, q_idx = np.random.choice(len(X_), size=2, replace=False)
            p = X_.pop(max(p_idx, q_idx)) # Remove o maior primeiro
            q = X_.pop(min(p_idx, q_idx))
            # Remove os pais de P_prime
            #print('len X_', len(X_))
            o_mask = p.copy()
            # Determina se será usada a primeira ou segunda lógica
            if np.random.rand() < 0.5:
                # Seleciona duas variáveis de decisão aleatórias de p.mask ∩ q.mask
                common_indices = np.where(p & ~q)[0]
                if len(common_indices) < 2:
                    O.append(np.array(o_mask))
                    continue  # Se não houver pelo menos dois elementos em comum, pula esta iteração
                m, n = np.random.choice(common_indices, size=2, replace=False)
                # Aplica a regra com base nas pontuações
                if self.sc[m] < self.sc[n]:
                    o_mask[m] = False
                else:
                    o_mask[n] = False
            else:
                # Seleciona duas variáveis de decisão aleatórias de p.mask ∩ q.mask
                common_indices = np.where(~p & q)[0]
                if len(common_indices) < 2:
                    O.append(np.array(o_mask))
                    continue  # Se não houver pelo menos dois elementos em comum, pula esta iteração
                
                m, n = np.random.choice(common_indices, size=2, replace=False)
    
                # Aplica a regra com base nas pontuações
                if self.sc[m] > self.sc[n]:
                    o_mask[m] = True
                else:
                    o_mask[n] = True
            # Adiciona o descendente à lista de descendentes
            O.append(np.array(o_mask))
        O = np.array(O)
        return O.reshape(2,int(O.shape[0]/2), O.shape[1])

class UniformCrossover(Crossover):

    def __init__(self, **kwargs):
        super().__init__(2, 2, **kwargs)

    def _do(self, _, X, **kwargs):
        _, n_matings, n_var = X.shape
        M = np.random.random((n_matings, n_var)) < 0.5
        _X = crossover_mask(X, M)
        return _X


class UX(UniformCrossover):
    pass

class ThreeToOneCrossover(Crossover):
    def __init__(self, n_matings=None, **kwargs):
        self.sc = kwargs.pop('sc', None)
        super().__init__(n_parents=3, n_offsprings=2, **kwargs)
        self.n_matings = n_matings  # Número de "matings" (ajustado para manter a população)

    def _do(self, problem, X, **kwargs):
        # Se n_matings não for definido, usa o tamanho da população original
        X_ = list(np.copy(X[0]))
        X_.extend(list(np.copy(X[1])))
        X = np.array(X_)
        n_matings = X.shape[0]
        offspring = np.full((n_matings, problem.n_var), False)

        for i in range(n_matings):
            # Seleciona 3 pais aleatórios (com substituição, se necessário)
            parent_indices = np.random.choice(X.shape[0], 3, replace=True)
            parents = X[parent_indices]
            child = self._create_child(parents)
            offspring[i, :] = child

        return offspring.reshape(2,int(offspring.shape[0]/2), offspring.shape[1])

    def _create_child(self, parents):
        p1, p2, p3 = parents[0], parents[1], parents[2]
        n_features = len(p1)
        child = np.full(n_features, False)

        # Lógica do crossover 3-to-1 (como implementado anteriormente)
        L1 = np.logical_and(p1, p2)
        L2 = np.logical_and(p1, p3)
        L3 = np.logical_and(p2, p3)
        S3 = L1 & L2 & L3
        S2 = (L1 | L2 | L3) & ~S3
        S1 = (p1 | p2 | p3) & ~S3 & ~S2
        child[L1 | L2 | L3] = True
        child[S3] = True
        S2_indices = np.where(S2)[0]
        S1_indices = np.where(S1)[0]

        if np.random.rand() < 0.5 and len(S2_indices) > 1:
            idx = np.random.choice(S2_indices, 2, replace= False)
            loser = idx[np.argmin(self.sc[idx])]
            child[loser] = False
        elif len(S1_indices) > 0:
            idx = np.random.choice(S1_indices, 2, replace=True if len(S1_indices) == 1 else False)
            winner = idx[np.argmax(self.sc[idx])]
            child[winner] = True
        return child

### Problem Difinition
class GeneSelection(ElementwiseProblem):
    def __init__(self, X, y, runner):
        self.n_features = X.shape[1]
        self.eval_dict = {'n_features':[], 'f1_score':[]}
        super().__init__(   n_var=self.n_features,
    						n_obj=2,
    						#n_constr=2,
    						xl=np.zeros(self.n_features),
    						xu=np.ones(self.n_features),
    						elementwise_evaluation=True,
                            type_var=bool,
                            save_history=True,
                            elementwise_runner=runner)

    def _evaluate(self, x, out, *args, **kwargs):
        selected_features = np.where(x == 1)[-1] # seleciona as features de acordo com o vetor binário
        X_selected = X_worker[:,selected_features]
        f_1 = []
        n_tests = 5
        seed = 41
        if len(selected_features) > 0:
            for i in range(n_tests):
                seed = seed + 1
                skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True) # Kfolding usado para separar em treino e teste
                clf = CLASSIFIER  # treino usando modelo SVM
                f_1.append(np.mean(cross_val_score(clf, X_selected, y_worker, cv=skf, scoring='f1_macro'))) # Computar f1
        
        f_1 = round(np.array(f_1).mean(), 3)
        n_features = len(selected_features)
        # salvar os resultados
        self.eval_dict['n_features'].append(n_features)
        self.eval_dict['f1_score'].append(f_1)
        out["F"] = [n_features, -f_1] # define os objetivos a serem minimizados
        # Restrições: limite mínimo e máximo de features selecionadas
        g1 = MIN_FEATURES - n_features  # Deve ser <= 0
        g2 = n_features - MAX_FEATURES  # Deve ser <= 0
        #out["G"] = np.column_stack([g1, g2])

def _init_evaluator(X, y):
    global pid_, X_worker, y_worker
    pid_ = os.getpid()
    X_worker = np.array(X).copy()
    y_worker = np.ravel(y).copy()

def run_experiment(**kwargs):
    problem = kwargs.get('problem', None)
    n_population = kwargs.get('n_population', 100)
    n_gen = kwargs.get('n_gen', 100)
    sampling = kwargs.get('sampling', None)
    seed = kwargs.get('seed', 42)
    crossover = kwargs.get('crossover', UniformCrossover())
    mutation = kwargs.get('mutation', None)
    max_features = kwargs.get('max_features', 100)
    verbose = kwargs.get('verbose', True)
    algorithm = kwargs.get('algorithm', None)
    
    algorithm = algorithm(pop_size = n_population,
                      sampling = sampling,
                	  crossover = crossover,
                	  mutation = mutation,
                      save_history = True)
    
    result = minimize(problem,  # problem class
                      algorithm,  # algorithm
                      ("n_gen", n_gen), # number of iteration for eval problem class
                      verbose=verbose)
    return result

def get_hv(result, **kwargs):
    n_evals = np.array([e.evaluator.n_eval for e in result.history])
    opt = np.array([e.opt[0].F for e in result.history])
    X_res, F_res = result.opt.get("X", "F")
            
    hist = result.history
    max = 100
    ref_point = np.array([15, -0.75])
    ind = HV(ref_point=ref_point)
            
    n_evals = []             # corresponding number of function evaluations\
    hist_F = []              # the objective space values in each generation
    hist_cv = []             # constraint violation in each generation
    hist_cv_avg = []         # average constraint violation in the whole population
            
    for algo in hist:
        n_evals.append(algo.evaluator.n_eval)  # store the number of function evaluations
        opt = algo.opt # retrieve the optimum from the algorithm
        # store the least contraint violation and the average in each population
        hist_cv.append(opt.get("CV").min())
        hist_cv_avg.append(algo.pop.get("CV").mean())
        # filter out only the feasible and append and objective space values
        feas = np.where(opt.get("feasible"))[0]
        hist_F.append(opt.get("F")[feas])
        
    metric = HV(ref_point= ref_point, norm_ref_point=False)
    hv = [metric.do(_F)/max for _F in hist_F]
    return n_evals, hv

def plot_convergence(results, **kwargs):
    color = kwargs.get('color', 'black')
    alpha = kwargs.get('alpha', 0.5)
    label = kwargs.get('label', None)
    hvs, n_evals = [], []
    for result in results:
        hv = get_hv(result)
        hvs.extend(hv[1])
        n_evals.extend(hv[0])
    df =  pd.DataFrame({
            "Function Evaluations": n_evals,
            "Hypervolume": hvs})

    sns.lineplot(data=df, x="Function Evaluations", y="Hypervolume", color = color, label=label, alpha = alpha).set_title('Convergence')

def plot_pareto_front(result, **kwargs):
    color = kwargs.get('color', 'black')
    #plt.figure(figsize=(7, 5))
    F_res = result.opt.get("F")
    plt.plot(F_res[:, 0][np.argsort(F_res[:, 1])], F_res[:, 1][np.argsort(F_res[:, 1])], color = color,  marker='o', mfc=color, mec=color, ms=2, ls='--', lw=0.5, zorder=2)
    plt.title("Objective Space")

def plot_multiple_pareto_front(results, **kwargs):
    color = kwargs.get('color', 'black')
    title = kwargs.get('title', 'Mean Pareto Front')
    label = kwargs.get('label', '-')
    alpha = kwargs.get('alpha', 0.7)
    moead = kwargs.get('moead', False)
    #plt.figure(figsize=(7, 5))
    F_res = []
    for result in results:
        F_res.extend(list(result.opt.get("F"))) 
    F_res = np.array(F_res)
    unique_keys = np.unique(F_res[:, 0])
    F_res = np.array([[key, F_res[F_res[:, 0] == key, 1].mean()] for key in unique_keys])
    
    df = pd.DataFrame({'n_features': F_res[:, 0][np.argsort(F_res[:, 0])],
                      'f1_score': F_res[:, 1][np.argsort(F_res[:, 0])]})
    if moead:
        df[['n_features']] = df[['n_features']] * 60
    sns.scatterplot(data = df ,x = 'n_features', y = 'f1_score', color = color,  marker='o', label=label, alpha = alpha)
    sns.lineplot(data = df ,x = 'n_features', y = 'f1_score', color = color, legend=False, linewidth=1, alpha = alpha)
    plt.title(title)
    plt.legend(loc="upper right")

def plot_best_pareto_front(results, **kwargs):
    color = kwargs.get('color', 'black')
    title = kwargs.get('title', 'Best Pareto Front')
    label = kwargs.get('label', '-')
    #plt.figure(figsize=(7, 5))
    F_res = []
    for result in results:
        F_res.extend(list(result.opt.get("F"))) 
    F_res = find_pareto_frontier(F_res)
    F_res = np.array(F_res)
    unique_keys = np.unique(F_res[:, 0])
    F_res = np.array([[key, F_res[F_res[:, 0] == key, 1].min()] for key in unique_keys])
    plt.plot(F_res[:, 0][np.argsort(F_res[:, 0])], F_res[:, 1][np.argsort(F_res[:, 0])], color = color,  marker='o', mec=color, ms=2, ls='--', lw=0.5, zorder=2, label=label)
    plt.title(title)
    plt.legend(loc="upper right")


def find_pareto_frontier(points):
    unique_points = []
    seen = set()
    for point in points:
        tuple_point = tuple(point)
        if tuple_point not in seen:
            seen.add(tuple_point)
            unique_points.append(point)
    frontier = []
    for p in unique_points:
        is_dominated = False
        for q in unique_points:
            if np.array_equal(q, p):
                continue  # Não comparar o mesmo ponto
            if (np.all(q <= p) and np.any(q < p)):
                is_dominated = True
                break
        if not is_dominated:
            frontier.append(p)
    
    return frontier

def getSparseEAWeight(X, y):
    X = np.array(X)
    weights = []
    for i in trange(len(X[0])):
        f_1 = []
        n_tests = 5
        seed = 41
        X_selected = X[:,i]
        for i in range(n_tests):
            seed = seed + 1
            skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True) # Kfolding usado para separar em treino e teste
            clf = CLASSIFIER # treino usando modelo SVM
            f_1.append(np.mean(cross_val_score(clf, X_selected.reshape(-1, 1), y, cv=skf, scoring='f1_macro', n_jobs = 10))) # Computar f1
        
        weights.append(round(np.array(f_1).sum()/n_tests, 3))
    return weights

def getRelieff(X, y):
    relieff = ReliefF(n_features_to_select=10, n_neighbors=10, n_jobs=N_PROCESS)  # Ajuste os parâmetros conforme necessário
    
    relieff.fit(np.array(X), np.ravel(np.array(y)))
    return relieff.feature_importances_

def getRF(X,y):
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=N_PROCESS)
    clf.fit(X, np.ravel(y))
    return clf.feature_importances_

def run_dataset(X, y, n_experiments, n_population, n_gen, max_features, n_process):
    pool = multiprocessing.Pool(n_process, initializer=_init_evaluator(X,y))
    runner = StarmapParallelization(pool.starmap)
    problem = GeneSelection(X,y, runner)

    sc_rf = getRF(X, y)
    sc_relieff = getRelieff(X, y)
    sc_sparseEA = getSparseEAWeight(X,y)
    
    results_dict = dict({'age_moea': [],
                         'nsga2': [],
                         'spea2':[],
                         'sparseEA':[],
                         'mofs_rfga': [],
                         'nsga2_weighted': []})
    for i in tqdm(range(n_experiments)):
        result = run_experiment(problem = problem,
                   algorithm = AGEMOEA,
                   n_population = n_population,
                   n_gen = n_gen,
                   sampling =  BinaryRandomSampling(max_features = max_features),
                   seed = i,
                   mutation = BitflipMutationLimitedBalanced(max_features = max_features),
                   crossover = UniformCrossover(),
                   max_features = max_features)
        results_dict['age_moea'].append(result)
        
        result = run_experiment(problem = problem,
                   algorithm = NSGA2,
                   n_population = n_population,
                   n_gen = n_gen,
                   sampling = BinaryRandomSampling(max_features = max_features),
                   seed = i,
                   mutation = BitflipMutationLimitedBalanced(max_features = max_features),
                   crossover = UniformCrossover(),
                   max_features = max_features)
        results_dict['nsga2'].append(result)
    
        result = run_experiment(problem = problem,
                   algorithm = SPEA2,
                   n_population = n_population,
                   n_gen = n_gen,
                   sampling = BinaryRandomSampling(max_features = max_features),
                   seed = i,
                   mutation = BitflipMutationLimitedBalanced(max_features = max_features),
                   crossover = UniformCrossover(),
                   max_features = max_features)
        results_dict['spea2'].append(result)

        result = run_experiment(problem = problem,
                   algorithm = NSGA2,
                   n_population = n_population,
                   n_gen = n_gen,
                   sampling =  SparseEASampling(sc = sc_sparseEA, max_features = max_features),
                   seed = i,
                   mutation = SparseEAMutation(weights = sc_sparseEA, max_features = max_features),
                   crossover = SparseEACrossover(sc = sc_sparseEA),
                   max_features = max_features)
        results_dict['sparseEA'].append(result)
    
        result = run_experiment(problem = problem,
                   algorithm = NSGA2,
                   n_population = n_population,
                   n_gen = n_gen,
                   sampling = ReliefFSampling(sc = sc_relieff,  max_features = max_features),
                   seed = i,
                   mutation = ReliefFMutation(sc = sc_relieff, max_features = max_features),
                   crossover = ThreeToOneCrossover(sc = sc_relieff),
                   max_features = max_features)
        results_dict['mofs_rfga'].append(result)

        result = run_experiment(problem = problem,
                   algorithm = NSGA2,
                   n_population = n_population,
                   n_gen = n_gen,
                   sampling = BinaryDistributedWeightsSampling(sampling_weights = sc_rf, max_features = max_features),
                   seed = i,
                   mutation = BitflipMutationLimitedBalanced(weights = sc_rf, max_features = max_features),
                   crossover = UniformCrossover(),
                   max_features = max_features)
        results_dict['nsga2_weighted'].append(result)
        for label in results_dict:
            for res in results_dict[label]:
                res = clean_result(res)
    return results_dict

def plot_convergence_and_pareto_front(results):
    colors = {'age_moea': 'black',
            'nsga2': 'blue',
            'spea2': 'gray',
            'sparseEA': 'yellow',
            'mofs_rfga': 'purple',
            'nsga2_weighted': 'red'}
    for label in results:
        plot_multiple_pareto_front(results[label], color = colors[label], label = label)
    plt.show()
    for label in results:
        plot_convergence(results[label], color = colors[label], label = label)
        
def clean_result(res):
    for pop in res.history:
        pop.archive = None
        pop.advance = None
        pop.callback = None
        pop.data = None
        pop.mating = None
        pop.display = None
        pop.pop = None
        pop.problem = None
        pop.output = None
        pop.pop_size = None
        pop.problem = None
        pop.repair = None
        pop.result = None
        pop.return_least_infeasible = None
        pop.run = None
        pop.save_history = None
        pop.seed = None
        pop.setup = None
        pop.survival = None
        pop.tell = None
        pop.termination = None
        pop.tournament_type = None
        pop.verbose = None
    return res