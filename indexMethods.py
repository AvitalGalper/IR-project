from inverted_index_gcp import *
import gzip
import csv
from io import BytesIO

def read_posting_list(inverted, w, indexName):
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
    client = storage.Client.from_service_account_json(key)
    return client.bucket('irprojectaon')

def read_index(bucket, path):
    print(path)
    blobreader = bucket.get_blob(path).open('rb')
    pcl = pickle.load(blobreader)
    blobreader.close()
    return pcl

def read_page_rank(bucket, path):
    blobreader = bucket.get_blob(path).open('rb')
    with gzip.open(blobreader, 'rb') as f:
        pcl = pd.read_csv(f)
    result_dict = {int(row[0]): float(row[1]) for row in pcl.values}
    blobreader.close()
    return result_dict

def BM_25(dl, textIndex):
    pass
def TitleNormalize(titleIndex):
    pass
def Mearge(bm25,titleSimilarity):
    pass
def sortPageRank(listToSort):
    pass