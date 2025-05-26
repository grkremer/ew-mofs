from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_openml
from functions import *

MIN_FEATURES = 1
MAX_FEATURES = 100
N_PROCESS = 64

def wrap_execution(X, y, name):
    print(name)
    test = run_dataset(X = X, y = y, n_experiments = 31, n_population = 100, n_gen = 100, max_features = 100, n_process = N_PROCESS)
    with open("Data/results/moo_hfs" + name + "_DT.pkl", "wb") as f:  # 'wb' = write in binary mode
        pickle.dump(test, f)
    del test
    print('----------------------------------------------------------------------')
    

def main():
    '''#P01
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
    X = np.array(breast_cancer_wisconsin_diagnostic.data.features)
    y = np.ravel(np.array(breast_cancer_wisconsin_diagnostic.data.targets))
    wrap_execution(X,y,'P01')

    #P02
    satellite = fetch_ucirepo(id=146)
    X = np.array(satellite.data.features)
    y = np.ravel(np.array(satellite.data.targets))
    wrap_execution(X,y,'P02')
    
    #P03
    optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80) 
    X = np.array(optical_recognition_of_handwritten_digits.data.features)
    y = np.ravel(np.array(optical_recognition_of_handwritten_digits.data.targets))
    wrap_execution(X,y,'P03')
    
    #P04
    madelon = fetch_openml(data_id = 1485, parser = 'auto')
    X = madelon.data
    y = madelon.target
    wrap_execution(X,y,'P04')
    
    #P05
    X = pd.read_csv('./Data/UJIIndoorLoc_validationData.csv').iloc[:, :520]
    y = pd.read_csv('./Data/UJIIndoorLoc_validationData.csv')['BUILDINGID']
    wrap_execution(X,y,'P05')

    #P06
    har = fetch_openml(data_id=1478, parser='auto')
    X = har.data
    y = har.target
    wrap_execution(X,y,'P06')
    
    #P07
    isolet5 = pd.read_csv('./Data/isolet5.data', header=None)
    X = np.array(isolet5.iloc[:,:-1])
    y = np.ravel(np.array(isolet5.iloc[:,-1]))
    wrap_execution(X,y,'P07')
    
    #P08
    # Carregar o dataset "Multiple Features" (ID no OpenML: 554)
    mfeat_factors = fetch_openml(data_id=12, parser='auto')
    mfeat_fourier = fetch_openml(data_id=14, parser='auto')
    mfeat_karhunen = fetch_openml(data_id=16, parser='auto')
    mfeat_morphological = fetch_openml(data_id=18, parser='auto')
    mfeat_pixel = fetch_openml(data_id=20, parser='auto')
    mfeat_zernike = fetch_openml(data_id=22, parser='auto')
    
    fac = np.array(mfeat_factors.data)
    fou = np.array(mfeat_fourier.data)
    kar = np.array(mfeat_karhunen.data)
    mor = np.array(mfeat_morphological.data)
    pix = np.array(mfeat_pixel.data)
    zer = np.array(mfeat_zernike.data)
    
    # Converter para DataFrame do pandas
    X = np.hstack((fac, fou, kar, pix, zer, mor))
    y = np.ravel(np.array(mfeat_factors.target))
    wrap_execution(X,y,'P08')
    
    #P09
    cnae = fetch_openml(data_id=1468, parser='auto')
    X = np.array(cnae.data)
    y = np.ravel(np.array(cnae.target))
    wrap_execution(X,y,'P09')

    #P10
    qsar_androgen_receptor = pd.read_csv('./Data/qsar_androgen_receptor.csv', sep = ';')
    y = np.array(qsar_androgen_receptor['positive'])
    X = np.array(qsar_androgen_receptor.drop(['positive'],axis = 1))
    wrap_execution(X,y,'P10')
    
    #P11
    # Carregar o dataset "Micromass" (ID no OpenML: 1508)
    micromass = fetch_openml(data_id=1515, parser='auto')
    X = np.array(micromass.data)
    y = np.ravel(np.array(micromass.target)).astype('int')
    wrap_execution(X,y,'P11')
    
    #P12
    colon = fetch_openml(data_id=45087, parser='auto')
    X = np.array(colon.data)
    y = np.ravel(np.array(colon.target))
    wrap_execution(X,y,'P12')
    
    #P13
    Liver_GSE14520 = pd.read_csv('Data/Liver_GSE14520_U133A.csv', sep = ',')
    X = np.array(Liver_GSE14520.drop(['samples', 'type'],axis = 1))
    y = np.array(Liver_GSE14520['type'])
    wrap_execution(X,y,'P13')
    
    #P14
    Leukemia_GSE28497 = pd.read_csv('Data/Leukemia_GSE28497.csv', sep = ',')
    X = np.array(Leukemia_GSE28497.drop(['samples', 'type'],axis = 1))
    y = np.array(Leukemia_GSE28497['type'])
    wrap_execution(X,y,'P14')
    
    #P15
    Breast_GSE70947 = pd.read_csv('Data/Breast_GSE70947.csv', sep = ',')
    X = np.array(Breast_GSE70947.drop(['type', 'samples'], axis = 1))
    y = np.array(Breast_GSE70947['type'])
    wrap_execution(X,y,'P15')
    
    #P16
    Colorectal_GSE44076 = pd.read_csv('Data/Colorectal_GSE44076.csv', sep = ',')
    X = np.array(Colorectal_GSE44076.drop(['samples', 'type'],axis = 1))
    y = np.array(Colorectal_GSE44076['type'])
    wrap_execution(X,y,'P16')
    
    #P17
    Renal_GSE53757 = pd.read_csv('Data/Renal_GSE53757.csv', sep = ',')
    X = np.array(Renal_GSE53757.drop(['samples', 'type'],axis = 1))
    y = np.array(Renal_GSE53757['type'])
    wrap_execution(X,y,'P17')
    
    #P18
    Breast_GSE45827 = pd.read_csv('Data/Breast_GSE45827.csv', sep = ',')
    X = np.array(Breast_GSE45827.drop(['samples', 'type'],axis = 1))
    y = np.array(Breast_GSE45827['type'])
    wrap_execution(X,y,'P18')
    
    #P19
    Colorectal_GSE21510 = pd.read_csv('Data/Colorectal_GSE21510.csv', sep = ',')
    X = np.array(Colorectal_GSE21510.drop(['samples', 'type'],axis = 1))
    y = np.array(Colorectal_GSE21510['type'])
    wrap_execution(X,y,'P19')
    
    #P20
    Brain_GSE50161 = pd.read_csv('Data/Brain_GSE50161.csv', sep = ',')
    X = np.array(Brain_GSE50161.drop(['samples', 'type'],axis = 1))
    y = np.array(Brain_GSE50161['type'])
    wrap_execution(X,y,'P20')'''
    #P07
    isolet5 = pd.read_csv('./Data/isolet5.data', header=None)
    X = np.array(isolet5.iloc[:,:-1])
    y = np.ravel(np.array(isolet5.iloc[:,-1]))
    wrap_execution(X,y,'P07')

    #P06
    har = fetch_openml(data_id=1478, parser='auto')
    X = har.data
    y = har.target
    wrap_execution(X,y,'P06')

if __name__ == '__main__':
    main()
