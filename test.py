import itertools
# def calculate_precision_at_k(relevant_docs, result_list, k):
#     if k <= 0:
#         return 0
#     result_list = result_list[:k]
#     intersection = set(result_list) & set(relevant_docs)
#     return len(intersection) / k

def calculate_precision_at_k(true_list, predicted_list, k): #nir
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    if len(predicted_list) == 0:
        return 0.0
    return round(len([1 for doc_id in predicted_list if doc_id in true_set]) / len(predicted_list), 3)
def average_precision(true_list, predicted_list, k=40): #nir
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i,doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions)+1) / (i+1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions)/len(precisions),3)

def recall_at_k(true_list, predicted_list, k): #nir
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    if len(true_set) < 1:
        return 1.0
    return round(len([1 for doc_id in predicted_list if doc_id in true_set]) / len(true_set), 3)


# def calculate_f1_at_k(relevant_docs, result_list, k):
#     precision = calculate_precision_at_k(result_list, relevant_docs, k)
#     recall = len(set(result_list) & set(relevant_docs)) / len(relevant_docs) if len(relevant_docs) > 0 else 0
#     if precision + recall == 0:
#         return 0
#     return 2 * (precision * recall) / (precision + recall)

def calculate_f1_at_k(true_list, predicted_list, k): #nir
    p = calculate_precision_at_k(true_list, predicted_list, k)
    r = recall_at_k(true_list, predicted_list, k)
    if p == 0.0 or r == 0.0:
        return 0.0
    return round(2.0 / (1.0/p + 1.0/r), 3)


def mean_average(lst):
    return sum(lst)/len(lst)


def results_quality(true_list, predicted_list): #nir
    p5 = calculate_precision_at_k(true_list, predicted_list, 5)
    f1_30 = calculate_f1_at_k(true_list, predicted_list, 30)
    if p5 == 0.0 or f1_30 == 0.0:
        return 0.0
    return round(2.0 / (1.0/p5 + 1.0/f1_30), 3)

def run_test(resLst, queryList, relevantDocs, times):
    print("\n")
    new = resLst
    # new = []
    # for lst in resLst:
    #     newLst = []
    #     for i in lst:
    #         newLst.append(str(i))
    #     new.append(newLst)

    precision_at_5_scores = []
    f1_at_30_scores = []
    results_quality_scores = []
    recallLst = []
    averagePercision30 = []
    for i in range(len(relevantDocs)):
        query = queryList[i]
        relevant_docs = relevantDocs[i]
        print("Query", i+1,":", query)
        precision = calculate_precision_at_k(relevant_docs, new[i], 5)
        print("Precision:", precision)
        precision_at_5_scores.append(precision)
        f1_at_30 = calculate_f1_at_k(relevant_docs, new[i], 30)
        print("F1:", f1_at_30)
        f1_at_30_scores.append(f1_at_30)

        results_Quality = results_quality(relevant_docs, new[i])
        print("Results Quality:", results_Quality)
        results_quality_scores.append(results_Quality)

        recall = recall_at_k(relevant_docs, new[i], 30)
        print("recall at 30:", recall)
        recallLst.append(recall)

        avgPer = average_precision(relevant_docs, new[i], 30)
        print("Average Precision:", avgPer)
        averagePercision30.append(avgPer)


    print("\n")
    print("*********************TOTAL**********************")
    print("--------------", len(relevantDocs),"Queries Checked-----------------")
    results_quality_avg = mean_average(results_quality_scores)
    print("\nResult quality Average-", results_quality_avg)
    recall_avg = mean_average(recallLst)
    print("\nRecall Average-", recall_avg)
    precision_average = mean_average(precision_at_5_scores)
    print("\nRegular Average Precision-", precision_average)
    f1_average  = mean_average(f1_at_30_scores)
    print("\nF1 Average-", f1_average)
    meanAP = mean_average(averagePercision30)
    print("\nMean Average Precision(MAP@30)-", meanAP)
    time_average = mean_average(times)
    print("\nTime Average-", time_average)

def check_strings_in_list(strings_to_check, target_list):
    found_strings = []
    for string in strings_to_check:
        if string in target_list:
            found_strings.append(string)
    ratio = len(found_strings) / len(strings_to_check)
    return ratio, found_strings, len(strings_to_check)

def equal_list_myRes_vs_RelevantDoc(dictionaryTitle, dictionaryBm25, strings_list):
    id_list = [str(key) for key in dictionaryTitle.keys()]
    ratio, found_strings, l = check_strings_in_list(strings_list, id_list)
    print("------Title Compare-----")
    print("dictionaryTitle:", dict(itertools.islice(dictionaryTitle.items(), 30)))
    print("\nRatio:",ratio)
    print("\nNumber Of Matches:", len(found_strings))
    print("Found Strings:", found_strings)
    print("\nTotal len:",l)
    print("------------------------")
    id_list = [str(key) for key in dictionaryBm25.keys()]
    ratio, found_strings, l = check_strings_in_list(strings_list, id_list)
    print("------bm25 Compare-----")
    print("dictionaryBm25:", dict(itertools.islice(dictionaryBm25.items(), 30)))
    print("\nRatio:",ratio)
    print("\nNumber Of Matches:", len(found_strings))
    print("Found Strings:", found_strings)
    print("\nTotal len:",l)
    print("------------------------")



# def calculate_recall_at_10(relevant_docs, retrieved_docs):
#     """
#     Calculate Recall@10.
#
#     Parameters:
#         relevant_docs (set or list): Set or list of relevant document IDs.
#         retrieved_docs (list): List of retrieved document IDs.
#
#     Returns:
#         recall_10 (float): Recall@10 value.
#     """
#     relevant_docs = set(relevant_docs)
#     retrieved_docs = retrieved_docs[:10]  # Consider only the top 10 retrieved documents
#     num_relevant_retrieved = len(relevant_docs.intersection(retrieved_docs))
#     recall_10 = num_relevant_retrieved / len(relevant_docs) if len(relevant_docs) > 0 else 0
#     return recall_10