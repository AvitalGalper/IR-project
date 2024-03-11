<div style="text-align: center;">
    <img src="backend/Wikipedia_Search.gif" style="height: auto; width: 200">
</div>

# IR-Project

The purpose of this project is to implement a search engine for the entire English Wikipedia corpus. Given a query, the system retrieves the most relevant documents. The results are based on body, title and anchor of articles combined with PageRank and PageView. The system is served by a Flask server deployed on a Google Cloud VM instance, supporting one HTTP requests on port 8080: search. The engine preprocesses the query by clearing and tokenizing. The engine searching the relevant documents in the inverted index using the BM25 algorithm and Binary similarity. The calculation also considers the page views and page rank of each document.

## Inverted Index Structure

- **term_total:** Posting list per term while building the index
- **posting_locs:** A dictionary, which maps each word to name of the correlated bin file - which contains the whole posting list, and to the offset of the posting list in the file.
- **df:** A dictionary, which maps each word to the document frequence of each word. The index also contain the bin files - which hold the binary data each word posting list.
- **posting_list:** A list of (file_name, offset) pairs.

## files that we used in our project:

- **DL_BIG.plk:** This pkl file contains a dictionary that connects each document ID with its corresponding length.
- **TitleDict.pkl:** Inside this pkl file, there's a dictionary that links each document ID with its title.
- **PageViews.pkl:** Stored in this pkl file is dictionary that associates each document ID with its respective page view count. This count reflects the number of times a web page was accessed, specifically pertaining to the month of August 2021, where over 10.7 million articles were viewed.
- **PageRank.pkl:** This csv file holds a dictionary that matches each document ID with its corresponding page rank value. This value quantifies the importance of a web page.

## Project Structure
### indexes
This directory contains needed files to create all the big indexes (on the whole corpus), which are finally used to build the engine.

### frontend
This directory contains the file search_fronted.py that run the flask app which recieves http requests for queries, and returns the results for them.

### backend
Inside this folder, you'll find the functionality responsible for reading the posting list, performing BM25 calculations, and implementing all the functions used by the Search_frontend. This functionality is distributed across three files: "indexMethods.py", "inverted_index_gcp.py", and "EnumPaths.py".
