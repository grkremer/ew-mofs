import os
import numṕy as np
def _init_evaluator():
    global pid_, X_worker, y_worker, colunas_worker
    pid_ = os.getpid()
    X_worker = np.array(X.copy())
    y_worker = np.array(y.copy())
    colunas_worker = colunas.copy()
    print(pid_)
def _init_mutator():
    global pid_, colunas_worker, matrix_similarity_worker, go_sets_labels_worker
    pid_ = os.getpid()
    colunas_worker = colunas.copy()
    matrix_similarity_worker = matrix_similarity.copy()
    go_sets_labels_worker = go_sets_labels.copy()