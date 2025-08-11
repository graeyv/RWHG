import numpy as np
import math
from src.helpers import sparsify_matrix

# safe divisions
epsilon = 1e-10

## --- function to compute a transition probability matrix P --- ##
def transition_prob_matrix(W):
    '''
    Computes a transition probability matrix given a weight matrix
    input: weight matrix (np array)
    output: transition probability matrix (np array)
        - Ex. P[1,2] tells us the prob. that random walker lands on node 2 next step when currently on node 1
    '''
    # handle potential NAs in weight matrices
    W = np.nan_to_num(W, nan=0.0)

    # compute degrees
    d = np.sum(W, axis=1) # col vector of rowwise sums of W (d[0] => sum of weights councillor 0 has with others)

    # safe divisions
    d[d == 0] = epsilon

    # compute inverse of degree matrices (straightforward for diagonal matrices)
    D_inv = np.diag(1 / d)

    # compute transition probability matrix P
    return D_inv @ W

## --- function to compute a relevance aka. steady state probability matrix --- ##
def steady_state_prob_matrix(P, W, alpha, priors=True): 
    '''
    Compute the steady state probability matrix (steady state distribution of Random Walk)
    input: 
        - P: transition probability matrix (np array)
        - W: weight matrix (np array)
        - alpha: restart probability of RW (teleports back to starting node)
        - priors: (boolean) when TRUE, the normalized Degree diagonal matrix F is used as priors in the model (captures fact that some politicians are more influential), otherwise I as in vanilla RWR
    output: s.s. probability aka. relevance matrix (np array)
        - Ex. R[1,2] tells us the steady state probability of walker landing on node 2 when walk has started on node 1.
    '''
    # compute degrees safely
    W = np.nan_to_num(W, nan=0.0)
    d = np.sum(W,axis=1)
    d[d == 0] = 1e-10

    # normalize degrees
    f = d / np.sum(d) # col vector of councillor degrees but normalized by overall degree

    # create prior matrix
    F = np.diag(f) # normalized version of D

    # create identity matrix
    I = np.eye(W.shape[0])

    # F = I is the general RW formulation; otherwise we include priors in the model reflecting influence of councillors, RW is more likely to restart at more influential nodes
    if priors is False: 
        F = I

    # compute relevance matrix R through steady state equation
    R = (1 - alpha) * np.linalg.inv(I - alpha * P) @ F

    return R

## --- function for iterative vote link prediction --- ##

def iterative_vote_link_prediction(P_yea_xy_t, P_nay_xy_t, R_x, R_y, W_yea_xy, W_nay_xy, targets, gamma,b, abstention): 
    '''
    inputs: 
        - matrices (as described already)
        - targets: np.array with 2 cols containing vote matrix index pairs (first col: row index, second col: col index)
        - gamma: parameter [0,1] weighting councillors vs affairs transition
        - b: batchsize. If b=1 then prediction is fully iterative, otherwise the b-top predictions get accepted for efficiency reasons
    output: 
        - filled vote weight matrices, indices and predictions
    '''

    # update transition probability matrices
    P_yea_xy_tplus1 = gamma * (R_x @ P_yea_xy_t) + (1 - gamma) * (P_yea_xy_t @ R_y)
    P_nay_xy_tplus1 = gamma * (R_x @ P_nay_xy_t) + (1 - gamma) * (P_nay_xy_t @ R_y)

    # compute prior probabilities for councillors and affairs
    p_prio_yea_x = np.sum(W_yea_xy, axis=1) / (np.sum(W_yea_xy, axis=1) + np.sum(W_nay_xy, axis=1) + epsilon) # vec with share of yes votes per councillor
    p_prio_nay_x =  np.sum(W_nay_xy, axis=1) / (np.sum(W_yea_xy, axis=1) + np.sum(W_nay_xy, axis=1) + epsilon) # share no votes

    p_prio_yea_y = np.sum(W_yea_xy, axis=0) / (np.sum(W_yea_xy, axis=0) + np.sum(W_nay_xy, axis=0) + epsilon)
    p_prio_nay_y = np.sum(W_nay_xy, axis=0) / (np.sum(W_yea_xy, axis=0) + np.sum(W_nay_xy, axis=0) + epsilon)

    # compute posterior probabilites for votes that should be predicted (for specific legislator vote pair)
    denominator = P_yea_xy_tplus1 + P_nay_xy_tplus1 + epsilon

    P_yea_ln = P_yea_xy_tplus1 / denominator # these are matrices of posterior probabilities (for each councillor-affair connection)
    P_nay_ln = P_nay_xy_tplus1 / denominator

    # approximate mutual information
    MI_right = (P_yea_ln + epsilon) * np.log((P_yea_ln + epsilon) / (p_prio_yea_x[:, None] * p_prio_yea_y[None, :] + epsilon))
    MI_left = (P_nay_ln + epsilon) * np.log((P_nay_ln + epsilon) / (p_prio_nay_x[:, None] * p_prio_nay_y[None, :] + epsilon))

    MI = MI_right + MI_left

    # extract only relevant predictions and find max link with max value
    row_indices = targets[:, 0]
    col_indices = targets[:, 1]
    MI_preds = MI[row_indices, col_indices]
    top_b_indices = np.argsort(MI_preds)[-b:][::-1] # when b=1: this is like np.argmax()
    l_star_arr = targets[top_b_indices, 0] # indices of councillor l (row)
    n_star_arr = targets[top_b_indices, 1] # indices of affair n (col) with highest MI 

    # update vote weight matrices according to rule
    preds = []

    # If we model abstentions as low yes or no probability
    if abstention: 
        for l_star, n_star in zip(l_star_arr, n_star_arr):
            # when both are just as likely => abstention
            if max(P_yea_ln[l_star, n_star], P_nay_ln[l_star, n_star]) <= 0.55:
                preds.append('EH')
            # when yes is more likely => yes
            if P_yea_ln[l_star, n_star] > P_nay_ln[l_star, n_star]:
                W_yea_xy[l_star, n_star] = 1
                preds.append('Yes')
            # when no is more likely => no
            elif P_yea_ln[l_star, n_star] < P_nay_ln[l_star, n_star]:
                W_nay_xy[l_star, n_star] = 1
                preds.append('No')
            else:
                continue


    # If we don't want to model abstentions
    else: 
        for l_star, n_star in zip(l_star_arr, n_star_arr):
            if P_yea_ln[l_star, n_star] > P_nay_ln[l_star, n_star]:
                W_yea_xy[l_star, n_star] = 1
                preds.append('Yes')
            elif P_yea_ln[l_star, n_star] < P_nay_ln[l_star, n_star]:
                W_nay_xy[l_star, n_star] = 1
                preds.append('No')
            else:
                continue

    return W_yea_xy, W_nay_xy, l_star_arr, n_star_arr, preds

## --- function which performs whole RWHG given vote matrices (votes, councillors, affairs) --- ##
def rwhg(W_yea_xy, W_nay_xy, W_x, W_y, test, b=1, gamma=6/7, alpha_x=0.85, alpha_y=0.85,K_x=6, K_y=6, abstention=False):
    '''
    function which executes the entire RWHG algorithm.
    inputs: 
        HYPERPARAMETERS
        - b: batch size for iterative prediction (b=1 predicts one value at a time)
        - gamma: weighting factor between political and semantic relevance path; higher => more emphasize on political path
        - alpha_x: (1 - alpha_x) represents the restart probability of the random walker for councillor graph
        - alpha_y: same for affairs graph
        - K_x: keep only K highest values in councillor weight matrix
        - K_y: same for affairs weight matrix (i.e. affairs graph)
        ADJACENCY MATRICES
        - W_yea_xy, W_nay_xy: Weight matrices from nay/yea network
        - W_x, W_y: Weigth matrices from councillor and affair network
        OTHER
        - test: 2D np.array with 2 columns (col0 contains the row indices of values to predict of W_yea_xy & W_nay_xy, while col1 contains the column indices)
    output: 
        - results: list of tupple (l_star_arr, n_star_arr, preds, truth) 
            - l_starr_arr: array with councillor (=row) indices in W_yea_xy / W_nay_xy that were predicted
            - n_starr_arr: array with affair (=column) indices in W_yea_xy / W_nay_xy that were predicted
            - preds: final prediction yes or no link for corresponding (l, n) pair in vote matrices
            - truth: what councillor l actually voted on affair n
            CAREFUL: l, n refer to the matrix indices not the ids, use mapping to get back to ids
    '''

    ## 1) Random Walk over unipartite Graphs (councillors & affairs)

    # apply knn to councillor and affair matrices
    W_x = sparsify_matrix(W_x,K_x)
    W_y = sparsify_matrix(W_y,K_y)

    # compute transition probability matrices 
    P_x = transition_prob_matrix(W_x)
    P_y = transition_prob_matrix(W_y)

    # compute relevance matrices
    R_x = steady_state_prob_matrix(P_x, W_x, alpha_x)
    R_y = steady_state_prob_matrix(P_y, W_y, alpha_y)

    ## 2) Random Walk over bipartite Graph (councillors vs affairs)

    # null out votes at test indices
    W_yea_xy[test[:, 0], test[:, 1]] = 0
    W_nay_xy[test[:, 0], test[:, 1]] = 0

    # compute initial transition probability matrices
    P_yea_xy_t = transition_prob_matrix(W_yea_xy)
    P_nay_xy_t = transition_prob_matrix(W_nay_xy)

    # for evaluation 
    results = []

    # loop parameters
    t = 0
    T = math.ceil(len(test) / b)

    # printing steps (10 approx. equally sized intervals)
    print_steps = set(np.linspace(1, T, num=min(10, T), dtype=int))

    # ensure batch size is not larger than test
    assert len(test) >= b, f"Batch size b={b} is too large for the test set of size {len(test)}"

    while True: 

        # update vote links and get indices of predicted vote link
        W_yea_xy, W_nay_xy, l_star_arr, n_star_arr, preds = iterative_vote_link_prediction(P_yea_xy_t, P_nay_xy_t, R_x, R_y, W_yea_xy, W_nay_xy, targets=test, gamma=gamma,b=b,abstention=abstention)

        # remove completed vote(s) from prediction targets (i.e. test set)
        ln_star_arr = np.column_stack((l_star_arr, n_star_arr))
        mask = ~np.any(np.all(test[:, None] == ln_star_arr, axis=2), axis=1)
        test = test[mask]

        # store results as tuples
        results.extend(zip(l_star_arr, n_star_arr, preds))

        if (t + 1) in print_steps:
            print(f"Iteration {t + 1} of {T}")

        # break once every target link has been predicted 
        t += 1
        if t == T:
            break

        # recompute intitial transition probability matrices with updated weight matrices
        P_yea_xy_t = transition_prob_matrix(W_yea_xy)
        P_nay_xy_t = transition_prob_matrix(W_nay_xy)

    return results