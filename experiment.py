from ucimlrepo import fetch_ucirepo 

from functions import *

MIN_FEATURES = 1
MAX_FEATURES = 100
N_PROCESS = 56

def wrap_execution(X, y, name):
    print(name)
    test = run_dataset(X = X, y = y, n_experiments = 5, n_population = 100, n_gen = 100, max_features = 100, n_process = N_PROCESS)
    with open("Data/results/" + name + ".pkl", "wb") as f:  # 'wb' = write in binary mode
        pickle.dump(test, f)
    del test
    print('----------------------------------------------------------------------')
    

def main():
    '''#DS01
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
    X = np.array(breast_cancer_wisconsin_diagnostic.data.features).astype('float32') 
    y = np.ravel(np.array(breast_cancer_wisconsin_diagnostic.data.targets))
    wrap_execution(X, y, 'breast_cancer_wisconsin_diagnostic')

    #DS02
    ionosphere = fetch_ucirepo(id=52)
    X = np.array(ionosphere.data.features).astype('float32') 
    y = np.ravel(np.array(ionosphere.data.targets))
    wrap_execution(X,y,'ionosphere')

    #DS06
    connectionist_bench_sonar_mines_vs_rocks = fetch_ucirepo(id=151) 
    X = np.array(connectionist_bench_sonar_mines_vs_rocks.data.features).astype('float32') 
    y = np.ravel(np.array(connectionist_bench_sonar_mines_vs_rocks.data.targets))
    wrap_execution(X,y,'connectionist_bench_sonar_mines_vs_rocks')

    #DS18
    qsar_androgen_receptor = pd.read_csv('./Data/qsar_androgen_receptor.csv', sep = ';') 
    y = np.array(qsar_androgen_receptor['positive'])
    X = np.array(qsar_androgen_receptor.drop(['positive'],axis = 1)).astype('float32')    
    wrap_execution(X,y,'qsar_androgen_receptor')

    #Cumida01
    data_Breast_GSE70947_norm = pd.read_pickle('Data/data_Breast_GSE70947_norm.pkl')
    X = np.array(data_Breast_GSE70947_norm.drop(['type'], axis = 1)).astype('float32')
    y = np.array(data_Breast_GSE70947_norm['type'])
    wrap_execution(X,y,'Breast_GSE70947_norm')'''

    #Cumida02
    Liver_GSE14520_U133A = pd.read_csv('Data/Liver_GSE14520_U133A.csv', sep = ',')
    X = np.array(Liver_GSE14520_U133A.drop(['samples', 'type'],axis = 1))
    y = np.array(Liver_GSE14520_U133A['type'])
    wrap_execution(X,y,'Liver_GSE14520_U133A_norm')
    

if __name__ == '__main__':
    main()
