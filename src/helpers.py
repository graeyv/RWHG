import numpy as np
from itertools import combinations
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import shap
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

## -- Function to compute gower weights -- ##
def get_gower_weights(councillors, features, target, model, s): 
    '''
    Computes weights as input for weighted Gower similarity function using different models
    Input: 
    - councillors:  pd.DataFrame of councillors
    - features:     features for which weights should be computed
    - target:       pd.DataFrame of pairwise co-sponsorship (CAREFUL TO TAKE CORRECT ONE)
    - model:        string to choose between 'logisitc', 'poisson', 'randomForestReg', 'randomForestClass'
    - s:            float between (0,1) which determines sample size for SHAP computation
    Output: 
    - pd.DataFrame with rounded weights & feature names as index
    '''
    # numerical per-variable similarity
    def numerical_per_var_similarity(col): 
        x = col.values
        s = 1 - (np.abs(x[:,None] - x[None, :])) / (np.nanmax(x) - np.nanmin(x)) 
        return s 
    
    # categorical per-variable similarity
    def categorical_per_var_similarity(col): 
        x = col.values
        s = (x[:, None] == x[None, :]).astype(float)
        return s  
    
    # initialize
    pairs = list(combinations(range(len(councillors)), 2))  # unique unordered pairs

    # Accumulate results here
    rows = []

    # Loop through features and compute similarities
    for feature in features:
        if np.issubdtype(councillors[feature].dtype, np.number):
            sim_matrix = numerical_per_var_similarity(councillors[feature])
        else:
            sim_matrix = categorical_per_var_similarity(councillors[feature])

        for i, j in pairs:
            rows.append({
                'elanId_1': councillors.iloc[i]['elanId'],
                'elanId_2': councillors.iloc[j]['elanId'],
                'feature': feature,
                'similarity': sim_matrix[i, j]
            })

    # Final long-format DataFrame
    similarity_df = pd.DataFrame(rows)

    # to wide format
    similarity_df_wide = similarity_df.pivot(
        index=['elanId_1', 'elanId_2'], 
        columns='feature', 
        values='similarity'
    ).reset_index()

    # inner-join with outcome
    df_merged = pd.merge(similarity_df_wide, target, how='inner', on=['elanId_1', 'elanId_2'])

    # create binary dummy of count cospon counts
    df_merged['dummy'] = (df_merged['count'] > 0).astype('int')

    # model
    if model == 'poisson' or model == 'logistic': 

        if model == 'poisson':
            # fit poisson regression
            mdl = smf.glm(formula='count ~ ' + ' + '.join(features), 
                            data=df_merged, 
                            family=sm.families.Poisson()).fit()
        else:
            # fit logistic regression
            mdl = smf.glm(formula='dummy ~ ' + ' + '.join(features),
                            data=df_merged,
                            family=sm.families.Binomial()).fit()

        # extract marginal effects
        marg_eff = mdl.get_margeff()
        marg_eff_df = marg_eff.summary_frame()
        dydx = marg_eff_df['dy/dx']

        # normalize weights
        v = dydx / dydx.sum() 

    elif model == 'randomForestReg' or model == 'randomForestClass':

        # features
        X = df_merged[features] 

        # random subsample to compute SHAP (efficiency)
        N = int(len(X) * s)
        sample_size = min(N, len(X)) 
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[sample_idx]

        if model == 'randomForestReg': 
            # fit RF Regressor
            y = df_merged['count']
            mdl = RandomForestRegressor(n_estimators=100, random_state=42)
            mdl.fit(X, y)

            # initialize SHAP explainer
            explainer = shap.TreeExplainer(mdl)
            shap_values = explainer.shap_values(X_sample)

            # compute mean absolute SHAP value per feature
            shap_importance = np.abs(shap_values).mean(axis=0)
            v = pd.Series(shap_importance, index=features)

            # normalize
            v = v / v.sum()

        else: 
            # fit RF Classifier
            y = df_merged['dummy']
            mdl = RandomForestClassifier(n_estimators=100, random_state=42)
            mdl.fit(X, y)

            # initialize SHAP explainer
            explainer = shap.TreeExplainer(mdl)
            shap_values = explainer.shap_values(X_sample)

            # compute mean absolute SHAP value per feature (for class 1)
            shap_values_class1 = np.abs(shap_values[:, :, 1])
            average_shap_class1 = np.mean(shap_values_class1, axis=0)
            v = pd.Series(average_shap_class1, index=features)

            # normalize
            v = v / v.sum()

    return round(v,4)


## -- Function to sparsify a weight matrix -- ##
def sparsify_matrix(W, k):
    '''
    Only keeps k largest values per row and nulls out rest
    Input: 
    - W: weight matrix (numpy array)
    - k: integer (amount of values to retain per row)
    Output: 
    - sparsified weight matrix (numpy array)
    '''
    n = W.shape[0]

    # k cannot be larger than matrix (minus diagonal)
    if k > n: 
        k = n - 1
        
    # Get indices of top-k values per row
    top_k_indices = np.argpartition(-W, kth=k, axis=1)[:, :k]

    # Create a mask of same shape, default False
    mask = np.zeros_like(W, dtype=bool)
    rows = np.arange(n)[:, None]  # Shape (n, 1) to broadcast
    mask[rows, top_k_indices] = True

    # Apply mask to keep top-k, zero the rest
    return W * mask


## -- Function to evaluate RWHG model -- ##
def evaluate_rwhg(results, ground_truth): 

    # convert results and ground truth to df
    df_pred = pd.DataFrame(results, columns=['c_idx', 'a_idx', 'pred'])
    df_true = pd.DataFrame(ground_truth, columns=['c_idx', 'a_idx', 'true'])

    # align
    df_eval = pd.merge(df_pred, df_true, on=['c_idx', 'a_idx'], how='inner')

    y_true = df_eval['true']
    y_pred = df_eval['pred']

    # infer sorted unique labels
    labels = sorted(set(y_true) | set(y_pred))

    # Per-class metrics
    precision = precision_score(y_true, y_pred, average=None, labels=labels)
    recall = recall_score(y_true, y_pred, average=None, labels=labels)
    f1 = f1_score(y_true, y_pred, average=None, labels=labels)

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # classification report
    print(classification_report(df_eval['true'], df_eval['pred'], output_dict=False))
    report = classification_report(df_eval['true'], df_eval['pred'], output_dict=True)
    weighted_avg_f1 = report['weighted avg']['f1-score']

    return accuracy, f1, precision, recall, weighted_avg_f1
