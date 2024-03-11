# IR-Project

The purpose of this project is to implement a search engine for the entire English Wikipedia corpus. Given a query, the system retrieves the most relevant documents. The results are based on body, title and anchor of articles combined with PageRank and PageView. The system is served by a Flask server deployed on a Google Cloud VM instance, supporting one HTTP requests on port 8080: search. The engine preprocesses the query by clearing and tokenizing. The engine searching the relevant documents in the inverted index using the BM25 algorithm and Binary similarity. The calculation also considers the page views and page rank of each document.

# Inverted Index Structure

_ term_total : Posting list per term while building the index
_ posting_locs: A dictionary, which maps each word to name of the correlated bin file - which contains the whole posting list, and to the offset of the posting list in the file.
_ df: A dictionary, which maps each word to the document frequence of each word. The index also contain the bin files - which hold the binary data each word posting list.
posting_list: A list of (file_name, offset) pairs.

# files that we used in our project:

DL_BIG.plk: pkl file contains a dictionary mapping document ID to his lenght.
TitleDict.pkl: pkl file contains a dictionary mapping document ID to the document's title.
pageviews.pkl: pkl file contains a dictionary mapping document ID to his page view, which is a value represents the request for the content of a web page (relevant to the month of August 2021, more then 10.7 million viewed atricles).
pagerank.pkl: file contains a dictionary mapping document ID to his page rank value, which is a value represent the importance of a web page.

# Project Structure
indexes
This folder contains needed files to create all the big indexes (on the whole corpus), which are finally used to build the engine.

frontend
This file contains the flask app which recieves http requests for queries, and returns the results for them.

backend
This file contains the functonlty of read the posting list etc, calculate BM25 and all the fuction Search_frontend use.
