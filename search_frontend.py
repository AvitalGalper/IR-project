import json

from flask import Flask, request, jsonify
from inverted_index_gcp import *
from indexMethods import *
from EnumPath import *
from time import time
from test import run_test, equal_list_myRes_vs_RelevantDoc


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        self.bucket = create_bucket(KEY)
        max_pagerank = 9913.72878216078
        # self.index_text_small = read_index(self.bucket, INDEX_TEXT+SEP+SMALL_TEXT_NAME_INDEX_PKL_FILE+PKL)
        # self.index_title_small = read_index(bucket, INDEX_TITLE+SEP+SMALL_TITLE_NAME_INDEX_PKL_FILE+PKL)
        # self.index_text_big = read_index(self.bucket, INDEX_TEXT+BIG_INDEX+SEP+BIG_TEXT_NAME_INDEX_PKL_FILE+PKL)
        self.index_title_big = read_index(self.bucket, INDEX_TITLE+BIG_INDEX+SEP+BIG_TITLE_NAME_INDEX_PKL_FILE+PKL)
        # self.index_title_big_stemming = read_index(self.bucket, INDEX_TITLE+BIG_INDEX+STEMMING+SEP+BIG_TITLE_NAME_INDEX_PKL_FILE_STEMMING+PKL)
        self.DL = read_index(self.bucket, DOC_LENGTH)
        self.ReadPageRank = read_page_rank(self.bucket, BIG_PAGE_RANK_PATH_CSV)
        self.PageRank = {doc_id: (score) / (max_pagerank) for doc_id, score in self.ReadPageRank.items()}
        self.index_text_big_with_filter = read_index(self.bucket, INDEX_TEXT+BIG_INDEX+BIG_INDEX_FILTER +SEP+BIG_TEXT_FILTER_NAME_INDEX_PKL_FILE+PKL)
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

def result_all_querys(query, indexQuery):
    avg_doc_length = 319.52423565619927
    t_start = time()
    print("\n")
    print("-----------------------------------", "Query *", indexQuery, "* :", query, "--------------------------------------")
    last_result, cosineScores, Bm25Scores = merge_results(query, app.index_title_big, app.index_text_big_with_filter, app.DL, avg_doc_length)
    result = merge_with_pagerank(last_result, app.PageRank)
    pr_time = time() - t_start
    return result, pr_time, cosineScores, Bm25Scores

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    # print_index_text_small = read_posting_list(app.index_text_small, "war", SMALL_TEXT_INDEX_NAME)
    # print("len",len(print_index_text_small))
    # print("\n")
    # print(print_index_text_small)
    # print_index_title_small = read_posting_list(app.index_title_small, "world", SMALL_TITLE_INDEX_NAME)
    # print(print_index_title_small)
    # print("Text BIG")
    # print_index_text_big = read_posting_list(app.index_text_big, "dog", BIG_TEXT_INDEX_NAME)
    # print("len", len(print_index_text_big))
    # print("\n")
    # print(sorted(print_index_text_big, key=lambda x: x[1], reverse=True))
    # print("\n\n\n\n\n\n\n")
    # print_index_text_big_with_filter = read_posting_list(app.index_text_big_with_filter, "dog", BIG_TEXT_FILTER_INDEX_NAME)
    # print("len", len(print_index_text_big_with_filter))
    # print("\n")
    # print(sorted(print_index_text_big_with_filter, key=lambda x: x[1], reverse=True))

    # print(print_index_text_big)
    # print("Title BIG")
    # print_index_title_big = read_posting_list(app.index_title_big, "game", BIG_TITLE_INDEX_NAME)
    # print(len(print_index_title_big))
    # print(print_index_title_big)
    # print_index_title_big_stemming = read_posting_list(app.index_title_big_stemming, "game", BIG_TITLE_INDEX_NAME_STEMMING)
    # print(len(print_index_title_big_stemming))
    # print(print_index_title_big_stemming)
    # print("Doc len for doc id 36443200")
    # print(app.DL[36443200])
    # print("Page Rank")
    # print(app.PageRank[3434750])  

    res = []
    # query = request.args.get('query', '')
    # if len(query) == 0:
    # #   return jsonify(res)
    # BEGIN SOLUTION
    # # avg_doc_length = sum(length for length in DL.values()) / len(DL)
    avg_doc_length = 319.52423565619927
    #
    # # sortWithPageRankScore = sortPageRank(meargeedList)
    # # res = sortWithPageRankScore
    # Open the JSON file
    PATH_TO_TRAIN_QUERIES = "C:\\Users\\avita\\OneDrive\\Desktop\\ir_proj_20240219\\queries_train.json"  # Update with the correct path

    with open(PATH_TO_TRAIN_QUERIES) as f:
        data = json.load(f)
    querys = []
    relevantDoc = []
    for q, relevantList in data.items():
        querys.append(q)
        relevantDoc.append(relevantList)

    resultList = []
    times = []
    indexOfQuery = 1
    listTitleScores = []
    listBM25Scores = []
    for q in querys:
        L, T, cosineScores, Bm25Scores = result_all_querys(q, indexOfQuery)
        indexOfQuery+=1
        resultList.append(L)
        times.append(T)
        listTitleScores.append(cosineScores)
        listBM25Scores.append(Bm25Scores)

    run_test(resultList,querys,relevantDoc, times)

    # ----------Test To Compare list of bm25 or list of title vs the docs in answers--------
    # QueryNumber = 14
    # equal_list_myRes_vs_RelevantDoc(listTitleScores[QueryNumber], listBM25Scores[QueryNumber], relevantDoc[QueryNumber])
    #END SOLUTION
    return jsonify(res)
    # return []
@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)



if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)

