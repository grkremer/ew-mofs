import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from pymoo.core.sampling import Sampling
from feature_selection_tools import *


def get_feature_mask_by_importance(
    values,
    n_samples,
    n_features,
    max_attempts=100,
    prob_multiplier=1.25,
    min_features=None,
):
    # Sort a value for each feature between df.min and df.max (from values).
    # Then select the top n_features that surpass this sorted value.
    # Adds some randomness while favoring top features.

    if min_features:
        size_constraint = min_features
    else:
        size_constraint = n_features

    if n_features >= values.shape[0]:
        raise Exception("Not enough features (n_features >= len(values))")

    selected_indexes = []
    selected_index_masks = []
    for sample_n in range(n_samples):
        selected_index = []
        attempts = 0

        # While selected_index is empty or another identical selected_index was already selected
        while len(selected_index) == 0 or np.any(
            [np.all(selected_index == item) for item in selected_indexes]
        ):
            selected_index = []
            while len(selected_index) < size_constraint:
                values_ = values.copy()
                # Set values for features already selected
                values_[selected_index] = np.min(values_)
                # Randomize values to select new candidate features (not selected yet)
                chances = np.random.uniform(
                    low=np.min(values_),
                    high=np.max(values_) * prob_multiplier,
                    size=values_.shape,
                )
                # Set chances for features already selected
                chances[selected_index] = np.min(values_)
                # Verify chosen features
                results = [
                    True if x >= y else False for (x, y) in zip(values_, chances)
                ]
                # Select top n_features
                selected_index = np.argwhere(results)[0:n_features, 0]

                # if min_features is set, select random final_n_features between min_features and n_features
                if min_features and len(selected_index) >= size_constraint:
                    upper_limit = min(n_features, len(selected_index))
                    final_n_features = (
                        np.random.randint(min_features, upper_limit)
                        if upper_limit > min_features
                        else min_features
                    )
                    selected_index = sorted(
                        np.random.choice(
                            selected_index, final_n_features, replace=False
                        )
                    )

                # if the feature set already exists in the collection, try flipping a bit to force diversion
                while np.any(
                    [np.all(selected_index == item) for item in selected_indexes]
                ):
                    # if min_features is set
                    if min_features and len(selected_index) >= size_constraint:
                        upper_limit = min(n_features, len(selected_index))
                        final_n_features = (
                            np.random.randint(min_features, upper_limit)
                            if upper_limit > min_features
                            else min_features
                        )
                        selected_index = sorted(
                            np.random.choice(
                                selected_index, final_n_features, replace=False
                            )
                        )
                    else:
                        final_n_features = n_features

                    # if the selected_index is present in the collection, try flipping a bit to force diversion
                    if (
                        np.any(
                            [
                                np.all(selected_index == item)
                                for item in selected_indexes
                            ]
                        )
                        and len(selected_index) >= size_constraint
                    ):
                        flip_position = np.random.choice(selected_index)
                        results[flip_position] = not results[flip_position]
                        selected_index = np.argwhere(results)[0:final_n_features, 0]

                    # Limit attempts to a maximum amount. Infinite loops are not yet defined.
                    attempts += 1
                    if attempts >= max_attempts:
                        raise Exception(
                            f"Attempts to generate new feature sets reached max_attempts ({max_attempts}). Amount of sets created: {len(selected_index_masks)}"
                        )

        top_n_selected_index = selected_index
        selected_indexes.append(top_n_selected_index)
        index_mask = np.full(values.shape[0], False, dtype=bool)
        index_mask[top_n_selected_index] = True
        selected_index_masks.append(index_mask)

    return np.array(selected_index_masks)

class FeatureSampling(Sampling):
    def __init__(self, feature_dfs = None, max_attempts=100, prob_multiplier=1.25, min_features=2):
        self.max_attempts = max_attempts
        self.prob_multiplier = prob_multiplier
        self.min_features = min_features
        self.feature_dfs = feature_dfs
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        # Load or generate feature importances. I think we could generate them before and just load them here.
        # Based on importances, we can set a chance of each feature being selected. Then generate feature sets randomly based on them.
        # We can specify which portion of n_samples will be generated by each method

        """
        feature_dfs = {...}
        fs_distrib = {"mrmr": 0.2, "relieff": 0.2, "kruskalwallis": 0.2, "mutualinfo": 0.2, "decisiontree": 0.2}
        """
        pool_values = self.feature_dfs
        #fs_distrib = problem.fs_distrib
        #methods = [method for method in fs_distrib]

        X = []
        for method in pool_values:

            X_ = get_feature_mask_by_importance(
                values=method,
                n_samples=int(n_samples/len(pool_values)),
                n_features=99,
                max_attempts=self.max_attempts,
                prob_multiplier=self.prob_multiplier,
                min_features=self.min_features,
            )
            #print(X_)
            # Reorder the resulting array - the feature_df does not use the same sorting as the original problem.df
            '''idx1 = np.argsort(problem.df.columns)
            idx2 = np.argsort(feature_df.index)
            idx1_inv = np.argsort(idx1)
            X_ = X_[:, idx2][:, idx1_inv]'''

            X.append(X_)

        X = np.concatenate(X)
        print(f"Generated {len(X)} new samples.")

        return X


def generate_n_importances(X_e, y_e, feature_names, dst_path, n, method_type="all"):
    if method_type == "all":
        methods_ = methods
    elif method_type == "deterministic":
        methods_ = deterministic_methods
    elif method_type == "non_deterministic":
        methods_ = non_deterministic_methods

    importances_dict = {}
    for method in methods_:
        # First pass of selections
        importances = method(X_e, y_e, feature_names, dst_path)

        # Reduce the original dataset to a smaller version with only n features
        X_selected, top_feature_names = select_top_n(X_e, feature_names, importances, n)

        # Perform feature selection again
        selected_importances = method(X_selected, y_e, top_feature_names, dst_path)

        # Set the discarded features to zero importance and recreate the series
        discarded_importances = importances[~importances.isin(selected_importances)]
        discarded_importances.index = np.zeros(len(discarded_importances))
        final_importances = pd.concat(
            [selected_importances, discarded_importances], axis=0
        )

        save_importances(
            final_importances.values,
            final_importances.index.values,
            method.__name__,
            dst_path,
            ascending=False,
        )
        importances_dict[method.__name__] = final_importances

    return importances_dict


def generate_n_degrees_importances(X_e, y_e, feature_names, dst_path, n, degrees):
    for method in methods:
        # First pass of selections
        importances = method(X_e, y_e, feature_names)

        X_selected = X_e
        top_feature_names = feature_names
        selected_importances = importances

        for n in degrees:
            # Reduce the original dataset to a smaller version with only n features
            X_selected, top_feature_names = select_top_n(
                X_selected, top_feature_names, selected_importances, n
            )

            # Perform feature selection again
            selected_importances = method(X_selected, y_e, top_feature_names)

        # Set the discarded features to zero importance and recreate the series
        discarded_importances = importances[~importances.isin(selected_importances)]
        discarded_importances.index = np.zeros(len(discarded_importances))
        final_importances = pd.concat(
            [selected_importances, discarded_importances], axis=0
        )

        save_importances(
            final_importances.values,
            final_importances.index.values,
            method.__name__,
            dst_path,
            ascending=False,
        )


def generate_importances(X_e, y_e, method_type="all"):
    """_summary_

    Args:
        X_e (_type_): Input feature dataset. Ideally, encoded.
        y_e (_type_): Input label dataset. Ideally, encoded.
        feature_names (_type_): List of feature names for the input feature dataset.
        dst_path (_type_): Path for the output files.
        method_type (str, optional): Types of methods to use. "all", "deterministic" or "non_deterministic". Defaults to "all".

    Returns:
        _type_: _description_
    """

    if method_type == "all":
        methods_ = methods
    elif method_type == "deterministic":
        methods_ = deterministic_methods
    elif method_type == "non_deterministic":
        methods_ = non_deterministic_methods
    importances_pool = []
    for method in methods_:
        importances_pool.append(method(X_e, y_e))
    return importances_pool


def save_importances(feature_names, importances, method, dst_path, ascending=False):
    # Save importances to a .csv file following the wTSNE format

    series = pd.Series(feature_names, index=importances).sort_index(ascending=ascending)
    pd.DataFrame(series, columns=["feature"]).reset_index().rename(
        columns={"index": "value"}
    )[["feature", "value"]].to_csv(
        f"{dst_path}/{method}_{len(feature_names)}_importances.csv", index=False
    )

    return series


def select_top_n(X, feature_names, importances, n):
    # Select top feature from the numpy array version of a dataset

    top_feature_names = importances.iloc[:n].values
    top_feature_indexes = np.array(
        [np.argwhere(feature_names == x) for x in top_feature_names]
    ).ravel()
    X_selected = X[:, top_feature_indexes]

    return X_selected, top_feature_names


def generate_relieff_importances(X_e, y_e):
    from skrebate import ReliefF

    method = "relieff"
    X_e = X_e.astype(np.float64)
    print(X_e)
    print(y_e)

    print(np.array(X_e))
    reliefF_clf = ReliefF(n_features_to_select=15, n_neighbors=100, n_jobs=-1).fit(
        np.array(X_e), y_e.ravel()
    )
    importances = reliefF_clf.feature_importances_

    return importances


def generate_kruskalwallis_importances(X_e, y_e):
    from scipy.stats import kruskal

    method = "kruskalwallis"
    try:
        label_groups = (X_e[np.argwhere(y_e == label)[:, 0]] for label in np.unique(y_e))
        res = kruskal(*label_groups)
        importances = (
            res.statistic
        )  # statistic, pvaluer for p-value (the closest to zero the the better)
        return importances
    except:
        return np.ones(X_e.shape[1], dtype=float)
    return np.ones(X_e.shape[1], dtype=float)


def generate_mutualinfo_importances(X_e, y_e):
    from sklearn.feature_selection import mutual_info_classif

    method = "mutualinfo"

    importances = mutual_info_classif(
        X_e,
        y_e.ravel(),
        n_neighbors=3,
    )
    
    return importances


def generate_lassocv_importances(X_e, y_e):
    from sklearn.linear_model import LassoCV
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    
    method = "lassocv"
    from sklearn.preprocessing import LabelEncoder
    enc = LabelEncoder()
    y_e = enc.fit_transform(y_e)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf = LassoCV().fit(X_e, y_e.ravel())
    importances = np.abs(clf.coef_)

    return importances


def generate_decisiontree_importances(X_e, y_e):
    from sklearn.tree import DecisionTreeClassifier

    method = "decisiontree"
    clf = DecisionTreeClassifier().fit(X_e, y_e.ravel())
    importances = clf.feature_importances_
    return importances


def generate_randomforest_importances(X_e, y_e):
    from sklearn.ensemble import RandomForestClassifier

    method = "randomforest"
    
    clf = RandomForestClassifier(n_estimators=150,).fit(X_e, y_e.ravel())

    importances = clf.feature_importances_

    return importances


def generate_linearsvm_importances(X_e, y_e):
    from sklearn.svm import LinearSVC

    method = "linearsvm"

    clf = LinearSVC(dual=True).fit(X_e, y_e.ravel())

    svm_weights = np.abs(clf.coef_).sum(axis=0)
    svm_weights /= svm_weights.sum()
    importances = svm_weights

    return importances


def generate_anovafvalue_importances(X_e, y_e):
    from sklearn.feature_selection import f_classif

    method = "anovafvalue"
    clf = f_classif(X_e, y_e.ravel())
    importances = clf[
        0
    ]  # [0] for statistic (the higher, the better), [1] for p-value (the closest to zero the the better)
    importances[np.isnan(np.array(importances))] = 0
    return importances


def generate_mrmr_importances(X_e, y_e, K=101):
    from mrmr import mrmr_classif
        # Converter para DataFrame com colunas no formato f0, f1, ...
    X_df = pd.DataFrame(X_e, columns=[f"f{i}" for i in range(X_e.shape[1])])
    y_series = pd.Series(y_e.ravel())
    
    # Definir K padrão como todas as features
    if K is None:
        K = X_e.shape[1]
    
    # Executar mRMR
    selected_features = mrmr_classif(X=X_df, y=y_series, K=K)
    
    # Extrair índices inteiros das features (ex: 'f2' → 2)
    selected_indices = [int(feature_name[1:]) for feature_name in selected_features]
    
    # Criar vetor de importância
    importances = np.zeros(X_e.shape[1])
    importances[selected_indices] = 1  # Usar índices inteiros
    
    return importances

deterministic_methods = [
    # Instant datasets (<60s)
    generate_kruskalwallis_importances,
    generate_anovafvalue_importances,
    # A few more minutes (>10min)
    generate_lassocv_importances,
    # A lot of minutes (>60min, +-3h for +-60k features)
    generate_relieff_importances,
    #generate_mrmr_importances,
]

non_deterministic_methods = [
    # Instant datasets (<60s)
    generate_decisiontree_importances,
    generate_randomforest_importances,
    generate_linearsvm_importances,
    # A few minutes (<10min)
    generate_mutualinfo_importances,
]

methods = deterministic_methods + non_deterministic_methods
