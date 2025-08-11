import numpy as np
import pandas as pd
import re
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


## --- Councillor Subgraph Approach I: co-sponsorship weight matrix --- ##
def cospon_weight_matrix(c_id2idx, affairs, councillors): 
    '''
    Computes the co-sponsorhip weight matrix (approach I for councillor subgraph)
    inputs: 
    - c_id2idx: dictionary that is a mapping of councillor ids to their corresponding index
    - affairs: pd DataFrame of affairs
    output: 
    - cosponsorship weight matrix W_x (numeric array)
    '''
    # councillor amount
    L = len(c_id2idx)

    # safe division
    epsilon = 1e-8

    # initialize count matrix
    C = np.zeros((L,L))

    # filter out affairs not authored by councillors (these are [])
    councillor_affairs = affairs[affairs['cosign_author_elanId'].apply(lambda x: len(x) > 0)]

    # fill matrix with cosponsorhsip counts
    for _, row in councillor_affairs.iterrows(): 

        # ensure that numpy arrays in each row contain unique ids which are int
        cosponsors = list(set([int(x) for x in row['cosign_author_elanId']]))

        # map elanIds in cosponsorship lists to matrix indices (if elanId is in councillors df)
        indices = [c_id2idx[x] for x in cosponsors if x in councillors['elanId'].values]

        # For each pair of co-sponsors, increment the corresponding entry in matrix C
        for i in range(len(indices)):
            for j in range(len(indices)):
                    C[indices[i], indices[j]] += 1 

    # transform count matrix to weight matrix according to formula
    W_x = C / (np.diag(C)[:, None] + np.diag(C)[None, :]+ epsilon)

    # remove self-connections from weight matrix (null out diagonal)
    np.fill_diagonal(W_x, 0) 

    # replace nan by 0
    W_x = np.nan_to_num(W_x, nan=0.0)  
            
    return W_x


## --- Councillor Subgraph Approach II: weighted Gower weight matrix --- ##
def weighted_gower_similarity(councillors, feature_cols, weights): 
    '''
    Computes weight matrix based on councillor features.
    Careful: assumes that ordinal variables are already encoded and does not handle NAs
    inputs: 
    - councillor df: pd Dataframe of councillors
    - feature cols: list of feature names
    - weights: list with weight with same order than features
    output: 
    - W_y: Gower similarity weight matrix
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

    # order data and select relevant variables
    councillors_ordered = councillors.sort_values(by='elanId')
    features = councillors_ordered[feature_cols]

    # identify column types
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    categorical_cols = features.select_dtypes(exclude=[np.number]).columns

    # intialize
    similarities = {}

    # compute per-var weighted similarity matrices
    for col in features.columns:
        
        v = weights[col]

        if col in categorical_cols:
            similarities[col] = categorical_per_var_similarity(features[col]) * v

        elif col in numeric_cols:
            similarities[col] = numerical_per_var_similarity(features[col]) * v

        else:
            raise ValueError(f"Column '{col}' is neither categorical nor numerical (or not specified).")    

    # Gower pairwise similarity matrix
    W_y = sum(similarities.values()) / weights[feature_cols].sum()

    # no self-similarity
    np.fill_diagonal(W_y, 0) 

    return W_y


## --- Affairs Subgraph Approach I: TF-IDF weight matrix --- ##
def tfidf_weight_matrix(affairs, ordered_a_ids): 
    '''
    Computes the TF-IDF weight matrix (Approach I for affairs subgraph)
    input:
    - affairs pd Dataframe
    - ordered set of affair ids
    output: 
    - weight matrix of affairs computed as cosine similarity of tf-idf matrix
    '''

    # order affairs according to array idx mapping
    affairs_ordered = affairs.set_index('id').loc[ordered_a_ids].reset_index()

    # convert topic arrays to space-separated strings
    topic_strings = affairs_ordered['topics'].apply(lambda x: " ".join(x) if isinstance(x, (list, np.ndarray)) else "")

    # Combine text fields and topics
    texts = (
        affairs_ordered[['title_text', 'submitted_text', 'reason_text']]
        .fillna("").astype(str).agg(" ".join, axis=1)
        + " " + topic_strings)

    # load german model from spacy for tokenizer and stopwords
    nlp = spacy.load("de_core_news_sm") 
    german_stopwords = set(stopwords.words('german'))

    # process text and lemmatize words
    def custom_lemmatizer(text):
        '''
        input: single string (aka. document)
        output: list of lemmatized tokens after processing
        '''
        text = text.lower()                         # Convert all characters to lowercase
        text = re.sub(r'[^\w\s]', '', text)         # Remove all punctuation (keep only words and spaces)
        text = re.sub(r'\d+', '', text)             # Remove all digits
        text = re.sub(r'\s+', ' ', text).strip()    # Replace multiple spaces with a single space and remove leading/trailing spaces

        doc = nlp(text)
        return [token.lemma_ for token in doc if token.lemma_ not in german_stopwords and not token.is_space]

    # TF-IDF vectorizer with custom tokenizer
    vectorizer = TfidfVectorizer(tokenizer=custom_lemmatizer, lowercase=False)  # lowercase is False because we do it ourselves
    tfidf_matrix = vectorizer.fit_transform(texts) # (1) tokenize each doc, (2) build vocabulary from all docs, (3) compute TF-IDF score for each doc

    # Compute weight matrix as cosine similarity
    W_y = cosine_similarity(tfidf_matrix)

    # remove self-connection in graph (null out diagonal)
    np.fill_diagonal(W_y, 0) 

    return W_y


## --- Affairs Subgraph Approach II: contextual embedding (jina-embeddings-v3) --- ##
def contextual_embedding_weight_matrix(affairs, ordered_a_ids):

    # order affairs according to array idx mapping
    affairs_ordered = affairs.set_index('id').loc[ordered_a_ids].reset_index()

    # function to combine fields by adding context
    def add_context(row):
        parts = []

        if row['title_text']:
            parts.append("Titel: " + row['title_text'])

        if isinstance(row['topics'], (list, np.ndarray)) and len(row['topics']) > 0:
            parts.append("Themen: " + ", ".join(row['topics']))

        if row['submitted_text']:
            parts.append("Eingereichter Text: " + row['submitted_text'])

        if row['reason_text']:
            parts.append("Begründung: " + row['reason_text'])

        return " ".join(parts)

    # Ensure all fields are in string format and missing values are handled
    affairs_ordered[['title_text', 'submitted_text', 'reason_text']] = (
    affairs_ordered[['title_text', 'submitted_text', 'reason_text']]
    .fillna("")
    .astype(str)
    )

    # Apply the context formatting row-wise
    texts = (affairs_ordered.apply(add_context, axis=1)  # use context function
            .str.replace(r'\s+', ' ', regex=True)       # collapse whitespaces to one
            .str.strip()                                # remove leading/trailing whitespaces
            )

    # load model
    model_name = "jinaai/jina-embeddings-v3"
    sts_model = SentenceTransformer(model_name, trust_remote_code=True)

    # Create embeddings for STS task in batch
    task = 'text-matching' # uses LoRa adapters for STS 
    embeddings = sts_model.encode(
        texts,
        batch_size=8,
        prompt_name=task,
        normalize_embeddings=True,
        show_progress_bar=False 
    )

    # Compute cosine similarity matrix
    W_y = cosine_similarity(embeddings)

    # remove self-connection in graph (null out diagonal)
    np.fill_diagonal(W_y, 0) 

    return W_y

## --- vote weight matrices (yea/nay matrices) --- ##
def yea_nay_weight_matrices(votes, c_id2idx, a_id2idx):

    # partition votes into Yes/No
    votes_yea = votes[votes['decision'] == 'Yes']
    votes_nay = votes[votes['decision'] == 'No']

    # initialize zero yea/nay weight matrices
    L = len(c_id2idx) # councillors
    N = len(a_id2idx) # affairs

    W_nay_xy = np.zeros((L,N))
    W_yea_xy = np.zeros((L,N))

    # fill 'nay' votes matrix (with consistent order)
    for _, row in votes_nay.iterrows():
        c_id = row['elanId']
        a_id = row['id']

        c_idx = c_id2idx[c_id]
        a_idx = a_id2idx[a_id]
        
        W_nay_xy[c_idx, a_idx] = 1

    # fill 'yea' votes matrix (with consistent order)
    for _, row in votes_yea.iterrows():
        c_id = row['elanId']
        a_id = row['id']

        c_idx = c_id2idx[c_id]
        a_idx = a_id2idx[a_id]

        W_yea_xy[c_idx, a_idx] = 1
    
    return W_yea_xy, W_nay_xy