import re
import nltk
from nltk.corpus import stopwords
import math
from inverted_index_gcp import *
import gzip
from EnumPath import *
import pandas as pd
nltk.download('stopwords')
from num2words import num2words

def read_posting_list(inverted, w, indexName):
    """
    This function reads the posting list for a given term from the inverted index.
    If the term is not found in the index, it returns an empty list.
    -----------
    Parameters:
    inverted: Inverted index containing posting lists.
    w: Term for which the posting list is to be retrieved.
    indexName: Name of the index file.
    -----------
    Returns:
    List of tuples representing the posting list for the given term, where each tuple contains a document ID and its term frequency.
    """
    if w not in inverted.posting_locs:
        return [] 
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, indexName)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
            tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
    return posting_list
    
def create_bucket(key):
    """
    This function connect to Google Cloud Storage bucket using the provided credentials.
    -----------
    Parameters:
    key: Path to the JSON key file for authentication. We dont need it for GCP.
    -----------
    Returns:
    Bucket object.
    """
    # client = storage.Client.from_service_account_json(key)
    client = storage.Client()
    bucket = client.bucket('irprojectaon')
    return bucket

def read_index(bucket, path):
    """
    This function reads an index stored in Google Cloud Storage.
    -----------
    Parameters:
    bucket: Bucket object representing the Google Cloud Storage bucket.
    path: Path to the index file within the bucket.
    -----------
    Returns:
    Index object loaded from the specified path.
    """
    blobreader = bucket.get_blob(path).open('rb')
    pcl = pickle.load(blobreader)
    blobreader.close()
    return pcl

def read_page_rank(bucket, path):
    """
    This function reads page rank data stored in Google Cloud Storage.
    -----------
    Parameters:
    bucket: Bucket object representing the Google Cloud Storage bucket.
    path: Path to the page rank file within the bucket.
    -----------
    Returns:
    Dictionary mapping document IDs to their corresponding page rank scores.
    """
    blobreader = bucket.get_blob(path).open('rb')
    with gzip.open(blobreader, 'rb') as f:
        pcl = pd.read_csv(f)
    result_dict = {int(row[0]): float(row[1]) for row in pcl.values}
    blobreader.close()
    return result_dict

def normalize_score(scores, min_val, max_val):
    """
    This function normalizes scores to a range between 0 and 1 based on the provided minimum and maximum values.
    If the input scores dictionary is empty or if the range of values (max_val - min_val) is zero, the function returns an empty dictionary.
    -----------
    Parameters:
    scores: Dictionary containing the scores to be normalized.
    min_val: Minimum value of the score range.
    max_val: Maximum value of the score range.
    -----------
    Returns:
    Dictionary containing the normalized scores with keys unchanged.
    """
    if not scores:
        return {}
    val_range = max_val - min_val
    if val_range == 0:
        return scores
    normalized_dict = {key: (value - min_val) / val_range for key, value in scores.items()}
    return normalized_dict


def calculate_bm25_score(query, index_big, DL, avg_doc_length, readPL, k1, b):
    """
    This function calculates the BM25 score for a given query and returns the top 1000 documents with their scores.
    -----------
    Parameters:
    query: List of terms in the query.
    index_big: Inverted index containing the posting lists for terms.
    DL: Dictionary containing the length of each document.
    avg_doc_length: Average length of documents in the collection.
    readPL: Index name for reading posting lists.
    k1: Parameter controlling term frequency saturation in BM25.
    b: Parameter controlling length normalization in BM25.
    -----------
    Returns:
    Dictionary containing the top 1000 documents and their BM25 scores.
    """
    minValue = 0
    maxValue = 0
    scores = defaultdict(float)
    corpus_len = 6348910
    idf = calculate_idf(query, index_big, corpus_len)
    for term in query:
        bigIndex = read_posting_list(index_big, term, readPL)
        if bigIndex == []:
            continue
        for i in range(len(bigIndex)):
            curr = bigIndex[i][0]
            if curr not in scores:
                scores[curr] = 0
            doc_length = DL.get(curr)
            if not doc_length:
                doc_length = avg_doc_length
            tf = bigIndex[i][1]
            current_score = (idf[term] * ((tf * (k1 + 1))) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length))))
            scores[curr] += current_score
            if current_score<minValue:
                minValue = current_score
            if current_score>maxValue:
                maxValue = current_score
    normalized_dict = normalize_score(scores, minValue, maxValue)
    return dict(Counter(normalized_dict).most_common(1000))

def calculate_idf(query, index_text_big, corpus_len):
    """
    This function calculates the inverse document frequency (IDF) for each term in the query.
    -----------
    Parameters:
    query: List of terms in the query.
    index_text_big: Inverted index containing document frequencies for terms.
    corpus_len: Total number of documents in the corpus.
    -----------
    Returns:
    Dictionary containing the IDF value for each term in the query.
    """
    idf = {}
    for term in query:
        if term in index_text_big.df:
            fterm = index_text_big.df[term]
            idf[term] = math.log(1 + (corpus_len - fterm + 0.5) / (fterm + 0.5))
    return idf

def similarity(query, index_big, readPL):
    """
    This function calculates the binary similarity scores between the query and documents in the index.
    -----------
    Parameters:
    query: List of tokens in the query.
    index_big: Inverted index of documents.
    readPL: Function to read posting lists.
    -----------
    Returns:
    similarities: Dictionary containing document IDs as keys and their binary similarity scores as values.
    """
    cosine_similarities = {}
    lenQuery = len(query)
    for term in query:
        bigIndex = read_posting_list(index_big, term, readPL)
        if bigIndex == []:
            continue
        for id,tf in bigIndex:
            if (id not in cosine_similarities.keys()):
                cosine_similarities[id] = 0
            cosine_similarities[id] += 1
    for id in cosine_similarities.keys():
        cosine_similarities[id] = cosine_similarities[id]*(1/lenQuery)
    return dict(Counter(cosine_similarities).most_common(1000))

def tokenize(text):
    """
    This function tokenizes a text into a list of tokens, handling special cases like Roman numerals and upper-case words.
    -----------
    Parameters:
    text: The text to tokenize.
    -----------
    Returns:
    list_of_tokens: A list of tokens extracted from the text, including handling of Roman numerals and upper-case words.
    upperWord: A boolean indicating whether the text contains upper-case words.
    """
    list_of_tokens = []
    upperWord = False
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["describe", "consider", "considered", "known", "following", "many", "however", "would", "became","category", "references", "also", "external", "links",
                            "may", "first", "see", "history", "people", "one", "two",
                            "part", "thumb", "including", "second"]
    all_stopwords = english_stopwords.union(corpus_stopwords)
    text = re.sub(r'\b[IVXLCDM]+\b', '', text)
    numbers = re.findall(r'\b\d+\b', text)
    for number in numbers:
        words = num2words(int(number), to='year').split()
        list_of_tokens.append(number)
        list_of_tokens.extend(words)
        upperWord = True
    pattern = r"(?<=\s)([A-Z][a-z']+)"
    matches = re.findall(pattern, text)
    if matches:
        upperWord = True
    RE_WORD2 = re.compile(r"""[\#\@\w](['\-](?!\w*['s]))?(3D|2D|4D)|\b\w{2,24}\b""", re.UNICODE)
    list_of_tokens.extend([token.group() for token in RE_WORD2.finditer(text.lower()) if token.group() not in all_stopwords])
    return list_of_tokens, upperWord


def merge_between_results(bm25_scores, cosineTitle, cosineAnchor, bm25_weight, cosineTitle_weight, cosineAnchor_weight):
    """
    This function merges scores from different retrieval methods with specified weights.
    -----------
    Parameters:
    bm25_scores: Dictionary of BM25 scores for documents.
    cosineTitle: Dictionary of cosine similarity scores based on title.
    cosineAnchor: Dictionary of cosine similarity scores based on anchor text.
    bm25_weight: Weight for BM25 scores.
    cosineTitle_weight: Weight for cosine similarity scores based on title.
    cosineAnchor_weight: Weight for cosine similarity scores based on anchor text.
    -----------
    Returns:
    merged_scores: Merged scores from different retrieval methods with specified weights.
    """
    merged_scores = Counter()
    for doc_id in set(bm25_scores) | set(cosineTitle) | set(cosineAnchor):
        merged_scores[doc_id] += bm25_scores.get(doc_id, 0) * bm25_weight
        merged_scores[doc_id] += cosineTitle.get(doc_id, 0) * cosineTitle_weight
        merged_scores[doc_id] += cosineAnchor.get(doc_id, 0) * cosineAnchor_weight
    return merged_scores

def merge_results(query, index_anchor_big,index_title_big, index_text_big, DL, avg_doc_length = 319.52423565619927, wText = 2.2, wTitle = 0.7, wAnchor = 0.8, k = 3.2, b = 0.08):
    """
    This function merges scores from different retrieval methods based on specified weights and parameters.
    -----------
    Parameters:
    query: Input query to search.
    index_anchor_big: Inverted index for anchor text.
    index_title_big: Inverted index for title text.
    index_text_big: Inverted index for main text.
    DL: Dictionary containing document lengths.
    avg_doc_length: Average document length.
    wText: Weight for BM25 scores based on main text.
    wTitle: Weight for cosine similarity scores based on title text.
    wAnchor: Weight for BM25 scores based on anchor text.
    k: Parameter for BM25 scoring.
    b: Parameter for BM25 scoring.
    -----------
    Returns:
    merge_between_result: Merged scores from different retrieval methods.
    """
    tokenize_query, upperWord = tokenize(query)
    print(tokenize_query)
    if len(tokenize_query) < 3:
        if len(tokenize_query) == 1:
            wTitle,wAnchor,wText,k,b = 1.3, 1.5, 2, 10, 0.0001
        else:
            wTitle,wAnchor,wText,k,b = 2, 1.8, 0.8, 10, 0.0001
    elif upperWord:
        if len(tokenize_query) == 3:
            wTitle,wAnchor,wText= 1, 2.5, 0.6
        else:
            wTitle,wAnchor,wText = 0.7, 0.8, 2.2
    cosine_scores_Anchor = calculate_bm25_score(tokenize_query, index_anchor_big, DL, avg_doc_length, BIG_ANCHOR_INDEX_NAME, k, b)
    cosine_scores_Title = similarity(tokenize_query, index_title_big, BIG_TITLE_INDEX_NAME)
    bm25_scores = calculate_bm25_score(tokenize_query, index_text_big, DL, avg_doc_length, BIG_TEXT_FILTER_INDEX_NAME, k, b)
    merge_between_result = merge_between_results(bm25_scores, cosine_scores_Title,cosine_scores_Anchor, wText, wTitle, wAnchor)
    return merge_between_result.most_common(750)

def normalize_pageRank(pagerank_scores):
    """
    This function normalizes page rank scores to a range between 0 and 1.
    -----------
    Parameters:
    pagerank_scores: Dictionary containing page rank scores for each document.
    -----------
    Returns:
    normalized_pagerank: Normalized page rank scores.
    """
    max_pagerank = 9913.72878216078
    normalized_pagerank = {doc_id: (score) / (max_pagerank) for doc_id, score in pagerank_scores.items()}
    return normalized_pagerank


def merge_with_pagerank_cut60(document_ids, pagerank_dict, pageview_dict, res_score = 0.5, pageRank_score = 0.7, pageView_score = 0.8):
    """
    This function merges search results with page rank and page view scores, and selects the top 60 documents based on the merged scores.
    -----------
    Parameters:
    document_ids: List of tuples containing document IDs and their corresponding scores.
    pagerank_dict: Dictionary containing page rank scores for each document.
    pageview_dict: Dictionary containing page view scores for each document.
    res_score: Score assigned to search results.
    pageRank_score: Score assigned to page rank.
    pageView_score: Score assigned to page views.
    -----------
    Returns:
    sorted_results: List of top 60 documents sorted by merged scores.
    """
    merged_results = {}
    for doc_id, score in document_ids:
        pagerank_score = pagerank_dict.get(doc_id, 0)
        pageview_score = pageview_dict.get(doc_id, 0)
        result_score = score
        merged_results[doc_id] = (result_score * res_score) + (pagerank_score * pageRank_score) + (pageview_score * pageView_score)
    sorted_results = sorted(merged_results.items(), key=lambda x: x[1], reverse=True)[:60]
    return sorted_results

def resultWithTitle(result_list, title_dict):
    """
    This function combines search results with corresponding titles.
    -----------
    Parameters:
    result_list: List of tuples containing document IDs and their corresponding scores.
    title_dict: Dictionary containing titles for each document ID.
    -----------
    Returns:
    result: List of tuples containing document IDs and their corresponding titles.
    """
    result = []
    for doc_id, score in result_list:
        if doc_id != 0:
            title = title_dict.get(doc_id)
            if title is not None: 
                result.append((doc_id, title))
    return result


# def calculate_cosine_similarity(query, document_vector):
#     dot_product = sum(x * y for x, y in zip(query, document_vector))
#     query_norm = math.sqrt(sum(x ** 2 for x in query))
#     doc_norm = math.sqrt(sum(y ** 2 for y in document_vector))
#     if query_norm == 0 or doc_norm == 0:
#         return 0
#     return dot_product / (query_norm * doc_norm)


# def stamming(tokenize_query):
#     ps = PorterStemmer()
#     tokens = [ps.stem(word) for word in tokenize_query]
#     return tokens

# def get_top_30_results(sorted_results):
#     top_results = dict(list(sorted_results.items())[:30])
#     return top_results


# def cosine_similarity(query_to_search, index, DL, normalize_DL):
#     len_query = len(query_to_search)
#     query_dict = Counter(query_to_search)
#     mone = Counter()
#     normalization_query = 0
#     sum_q = 0
#     corpus_len = 6348910
#     for term in query_to_search:
#         bigIndex = read_posting_list(index, term, BIG_ANCHOR_INDEX_NAME)
#         if bigIndex == []:
#             continue
#         df = DL.get(term, 1)  # Get document frequency (DF) from DL dictionary, default to 1 if term not found
#         idf = math.log(len(normalize_DL) / df, 10)  # Calculate IDF using document frequency (DF)
#         for doc_id, frequency in bigIndex:
#             mone[doc_id] += frequency * query_dict[term] * idf
#         sum_q += (query_dict[term] * idf) ** 2

#     normalization_query = 1 / math.sqrt(sum_q)
#     for doc_id in mone.keys():
#         nf = 1 / math.sqrt(normalize_DL[doc_id])
#         mone[doc_id] *= normalization_query * nf
#     sorted_items = sorted(mone.items(), key=lambda item: item[1], reverse=True)
#     print(dict(sorted_items[:40]))
#     return dict(sorted_items[:40])


# from gensim.models import KeyedVectors
# import gensim.downloader as api
# wv = api.load('glove-wiki-gigaword-300')

# def QueryExpand(query, limiter=3):
#     expansion = []
#     for token in query:
#         try:
#             similar_words = wv.most_similar(token, topn=5)
#             expansion.extend([word for word, _ in similar_words])
#         except:
#             print("Word not in w2v Dictionary")
#     expansion = list(set(expansion)) 
#     expansion.sort(key=lambda x: x[1], reverse=True)
#     if len(expansion) >= limiter:
#         return expansion[:limiter]
#     return expansion   