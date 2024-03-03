import re
from nltk.corpus import stopwords
import math
from inverted_index_gcp import *
import gzip
from EnumPath import *
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from time import time

def read_posting_list(inverted, w, indexName):
    start_time = time()
    if w not in inverted.posting_locs:
        return []  # Return an empty list if the word is not found
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, indexName)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
            tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
    end_time = time()
    print(f"read_posting_list execution time: {end_time - start_time} seconds")
    return posting_list
    
def create_bucket(key):
    client = storage.Client.from_service_account_json(key)
    bucket = client.bucket('irprojectaon')
    return bucket

def read_index(bucket, path):
    print(path)
    blobreader = bucket.get_blob(path).open('rb')
    pcl = pickle.load(blobreader)
    blobreader.close()
    return pcl

def read_page_rank(bucket, path):
    print(path)
    blobreader = bucket.get_blob(path).open('rb')
    with gzip.open(blobreader, 'rb') as f:
        pcl = pd.read_csv(f)
    result_dict = {int(row[0]): float(row[1]) for row in pcl.values}
    blobreader.close()
    return result_dict

def normalize_bm25(scores):
    if not scores:
        return {}
    min_val = min(scores.values())
    max_val = max(scores.values())
    val_range = max_val - min_val
    if val_range == 0:
        return scores
    normalized_dict = {key: (value - min_val) / val_range for key, value in scores.items()}
    return normalized_dict


def calculate_bm25_score(query, index_text_big, DL, avg_doc_length):
    start_time = time()
    k1 = 3
    b = 0.15
    # if len(query) < 3:
    #     k1 = 3
    #     b = 0.15
    # else:
    #     k1 = 3
    #     b = 0.6    
    bm25_scores = {}
    corpus_len = 6348910
    idf = calculate_idf(query, index_text_big, corpus_len)
    for term in query:
        bigIndex = read_posting_list(index_text_big, term, BIG_TEXT_FILTER_INDEX_NAME)
        if bigIndex == []:
            continue
        for i in range(len(bigIndex)):
            curr = bigIndex[i][0]
            if curr not in bm25_scores:
                bm25_scores[curr] = 0
            bm25_score = 0
            doc_length = DL.get(curr)
            tf = bigIndex[i][1]
            bm25_score = (idf[term] * ((tf * (k1 + 1))) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length))))
            bm25_scores[curr] += bm25_score
    normalized_dict = normalize_bm25(bm25_scores)
    last_dict = dict(sorted(normalized_dict.items(), key=lambda x: x[1], reverse=True))
    end_time = time()
    print(f"calculate_bm25_score execution time: {end_time - start_time} seconds")
    return last_dict


def calculate_idf(query, index_text_big, corpus_len):
    idf = {}
    for term in query:
        if term in index_text_big.df.keys():
            fterm = index_text_big.df[term]
            idf[term] = math.log(1 + (corpus_len - fterm + 0.5) / (fterm + 0.5))
    return idf


def tokenize(text):
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = [ "following", "many", "however", "would", "became","category", "references", "also", "external", "links",
                            "may", "first", "see", "history", "people", "one", "two",
                            "part", "thumb", "including", "second"]
    all_stopwords = english_stopwords.union(corpus_stopwords)
    # RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    RE_WORD = re.compile(r"""[\#\@\w](['\-](?!\w*['s]))?\w{2,24}""", re.UNICODE)
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    return list_of_tokens


def calculate_cosine_similarity(query, document_vector):
    start_time = time()
    dot_product = sum(x * y for x, y in zip(query, document_vector))
    query_norm = math.sqrt(sum(x ** 2 for x in query))
    doc_norm = math.sqrt(sum(y ** 2 for y in document_vector))
    if query_norm == 0 or doc_norm == 0:
        return 0
    end_time = time()
    print(f"calculate_cosine_similarity execution time: {end_time - start_time} seconds")
    return dot_product / (query_norm * doc_norm)


def cosine_similarity_for_title(query, index_title_big):
    start_time = time()
    cosine_similarities = {}
    for term in query:
        bigIndex = read_posting_list(index_title_big, term, BIG_TITLE_INDEX_NAME)
        if bigIndex == []:
            continue
        for id,tf in bigIndex:
            if (id not in cosine_similarities.keys()):
                cosine_similarities[id] = 0
            cosine_similarities[id] +=1        
    for id in cosine_similarities.keys():
        cosine_similarities[id] = cosine_similarities[id]*(1/len(query))
    cosine_similarities = {k: v for k,v in sorted(cosine_similarities.items(),key=lambda x: x[1], reverse = True)}
    end_time = time()
    print(f"cosine_similarity_for_title execution time: {end_time - start_time} seconds")
    return cosine_similarities



def merge_between_results(bm25_scores, cosine_similarity, cosine_weight = 0.6, bm25_weight = 0.4):
    merged_scores = {}
    for doc_id in cosine_similarity.keys():
        cosine_score = cosine_similarity.get(doc_id, 0)  # Default value of 0 if key doesn't exist
        bm25_score = bm25_scores.get(doc_id, 0)      # Default value of 0 if key doesn't exist
        merged_scores[doc_id] = (cosine_weight * cosine_score) + (bm25_weight * bm25_score)
    return merged_scores

def merge_results(query, index_title_big, index_text_big, DL, avg_doc_length):
    start_time = time()
    tokenize_query = tokenize(query)
    print("tokenize Qurey:", tokenize_query)
    # lem_query = lemmatize_query(tokenize_query)
    cosine_scores = cosine_similarity_for_title(tokenize_query, index_title_big)
    # print("cosin scores:\n",get_top_30_results(cosine_scores))
    bm25_scores = calculate_bm25_score(tokenize_query, index_text_big, DL, avg_doc_length)
    # print("bm25 scores:\n",get_top_30_results(bm25_scores))
    # get_weight_by_query = adjust_weights_based_on_query(tokenize_query)
    merge_between_result = merge_between_results(cosine_scores, bm25_scores)
    sorted_results = dict(sorted(merge_between_result.items(), key=lambda x: x[1], reverse=True))
    top_30_results = get_top_30_results(sorted_results)
    end_time = time()
    print(f"\n TIME: {end_time - start_time} seconds")
    return top_30_results, cosine_scores, bm25_scores

def get_top_30_results(sorted_results):
    top_results = dict(list(sorted_results.items())[:30])
    return top_results

def normalize_pageRank(pagerank_scores):
    max_pagerank = 9913.72878216078
    normalized_pagerank = {doc_id: (score) / (max_pagerank) for doc_id, score in pagerank_scores.items()}
    return normalized_pagerank


def merge_with_pagerank(document_ids, pagerank_dict, res_score = 0.2, pageRank_score = 0.8):
    pagerank_subset = {doc_id: score for doc_id, score in pagerank_dict.items() if doc_id in document_ids}
    # normalized_pagerank = normalize_pageRank(pagerank_subset)
    merged_results = {}
    for doc_id in document_ids:
        pagerank_score = pagerank_subset.get(doc_id, 0)  # Get the PageRank score for the document
        result_score = document_ids.get(doc_id, 0)
        merged_results[doc_id] = (result_score * res_score) + (pagerank_score * pageRank_score)
    sorted_results = sorted(merged_results.items(), key=lambda x: x[1], reverse=True)
    return [tup[0] for tup in sorted_results]

# def adjust_weights_based_on_query(query):
#     start_time = time()
#     weight_text = 0.5
#     weight_title = 0.5
#     if len(query) == 0:
#         return weight_text, weight_title
#     x_percent = 0.2  # Default weight percentage for text retrieval
#     y_percent = 0.2  # Default weight percentage for title retrieval
    
#     if len(query) >= 3:
#         x_percent = 0.8  # Higher weight percentage for text retrieval
#     else:
#         y_percent = 0.8  # Higher weight percentage for title retrieval
#     weight_text = x_percent 
#     weight_title = y_percent 
#     end_time = time()
#     print(f"adjust_weights_based_on_query execution time: {end_time - start_time} seconds")
#     return weight_title, weight_text

# def lemmatize_query(query):
#     start_time = time()
#     lemmatizer = WordNetLemmatizer()
#     lemmatized_words = [lemmatizer.lemmatize(word, wordnet.VERB) for word in query]
#     end_time = time()
#     print(f"lemmatize_query execution time: {end_time - start_time} seconds")
#     return lemmatized_words
