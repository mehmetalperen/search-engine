import json
import shelve
import time

from sympy import python
from nltk.stem import PorterStemmer  # to stem
from stop_words import get_stop_words
import os
stop_words = set(get_stop_words('en'))


def tokenizer(page_text_content):
    '''
    tokenizer takes in a string, tokenizes it,
    and stems the tokens.
    Returns: (string) - list of stemmed tokens
    '''
    try:
        tokens = []
        cur_word = ""
        stemmer = PorterStemmer()
        for ch in page_text_content:  # read line character by character
            # pig's --> pigs is this a problem?
            # check if character is in english alphabet or a number
            if ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.-'":
                if ch in ".-'":
                    continue
                cur_word += ch.lower()  # convert that ch to lower case and add it to the cur_word
            elif len(cur_word) > 0:
                tokens.append(stemmer.stem(cur_word))
                cur_word = ""
        if len(cur_word) > 1:
            tokens.append(stemmer.stem(cur_word))
    except Exception as e:
        print(f"Error tokenizer: {str(e)}")
        return []
    # stemmed_tokens = [stemmer.stem(token) for token in tokens]

    return tokens


def handle_stopwords(query_tokens):
    '''
    if stopwords needed, keeps the original query_tokens and returns query_tokens 
    if stopwords NOT needed, removes the stopwords and returns query_tokens_no_stop_words
    '''
    global stop_words

    query_tokens_no_stop_words = [] # query after stop words removed
    removed = [] #list to store the removed stop words
    for token in query_tokens: 
        if token in stop_words: 
            removed.append(token) 
        else:
            query_tokens_no_stop_words.append(token)

    loss_percentange = 100 - ((len(query_tokens_no_stop_words)
                               * 100) / len(query_tokens)) # how much of the query is lost after removing the stop words

    if loss_percentange >= 50:  # more than 50 percent of the query is lost. stop words needed
        return query_tokens # return the original query 
    return query_tokens_no_stop_words  # stop words NOT needed, return list with NO stop words


def read_large_line(file):
    '''
    read_large_line reads the full line in a string no matter the size of the line
    - useful because we ran into a buffer overload error using the regular readline
      which has a default size of 4096
    Returns: the line (str)
    '''
    chunk_size = 4096  # python line buffer size
    line = ''

    #keep reading until it reaches a newline or EOF
    while True:
        chunk = file.readline(chunk_size)
        line += chunk

        if len(chunk) < chunk_size or '\n' in chunk:
            break

    return line

def boolean_and_search(boolean_query_list):
 
    try:
        # {'decemb': [[4, [1826, 1917], 2]]} => {word: [[docID, [positions...], tfidf]]}
        if len(boolean_query_list) == 0:
            return [] #return the empty list
        
        least_seen_word_object = boolean_query_list[0] # => {'decemb': [[4, [1826, 1917], 2]]}

        least_seen_word = list(boolean_query_list[0].keys())[0]  # 'decemb'

        baseList = [] #baseList is the list of docID's that are in the least_seen_word
        
        for posting in least_seen_word_object[least_seen_word]: #posting = [4, [1826, 1917], 2]
                baseList.append([posting[0], posting[2]]) #docID = posting[0], tf-idf = posting[2]

        if len(boolean_query_list) == 1: #if its 1 query term, then write out the documents that have that term
            return baseList


        minTFIDF = 0
        firstFewCount = 150
        
        for curTerm in boolean_query_list: #for every term in the query list
            for token in curTerm: #should only be 1
                i = 0
                j = 0
                while i < len(baseList): # for every element in the baselist

                    #calculate new minimum tf-idf to look for up until the "firstFewCount" runs out
                    if firstFewCount > 0:
                        firstFewCount -= 1
                        if minTFIDF > baseList[i][1]: #compare current minimum with the base list
                            minTFIDF = baseList[i][1]
                    else:
                        if minTFIDF > baseList[i][1]:
                            print("deletes...")
                            del baseList[i]
                            continue

                    while j < len(curTerm[token]): # for every posting in the current term
                        curPosting = curTerm[token][j] #note: curTerm[token][j] gives a posting, curPosting[0] gives the docID
                        if curPosting[0] > baseList[i][0]: #if current posting's doc id is greater than the baseList's doc id
                            del baseList[i]
                            i -= 1 #decriment it because it will be incrimented later (an incriment + a delition will make it go forward two spots)
                            if len(baseList) == 0:
                                return baseList #return empty list if empty
                            else:
                                break #go to the next element in baselist (with the i+= below), keep current position in curTerm[token]
                        elif curPosting[0] == baseList[i][0]: #if current posting has the same docID, add it to the list and add to baseList docID's tf-idf
                            baseList[i][1] += curPosting[2]
                            j += 1 #go to the next element in baselist (with the i+= below), incriment to next position in curTerm[token]
                            break
                        j += 1 #if nothing above, just incriment j to go to the next posting

                    i += 1

        baseList.sort(key = lambda x: x[1], reverse = True) #sort according to tfIDF
        if len(baseList) > 500: #get the top 200 documents
            baseList = baseList[:500]
        return baseList

                        

    except Exception as e:
        print('ERROR in generate_boolean_search_result: ', str(e))
        return []



def nGram_result(query_docsID_tfidf, boolean_query_list): #query term list, query document result
    '''returns a list of lists [[docID-1, tfidf-1], [docID-2, tfidf-2]] in order of decreasing tfidf'''
    doc_ngram_count = []
    intesection_docID = set()
    for docID_tfidf in query_docsID_tfidf:
        docID = docID_tfidf[0]
        tfidf = docID_tfidf[1]
        partial_ngrams = nGramDoc(docID, boolean_query_list) #returns an int of the number of partial ngrams found
        doc_ngram_count.append([docID, partial_ngrams + tfidf])
        intesection_docID.add(docID)
    
    doc_ngram_count = sorted(doc_ngram_count, key = lambda x: x[1], reverse = True)
    return (doc_ngram_count, intesection_docID)


#returns an int of the number of partial ngrams found
def nGramDoc(docID, boolean_query_list):
    #print("nGramDoc:")
    tokenPosInDoc = [] #token positions will be a list of lists (one list of positions for each term in the document)

    tfidfScore = 0
    for posting_list in boolean_query_list:
        found = False
        for token in posting_list: #should only be 1
            for posting in posting_list[token]:
                if posting[0] == docID: #when you find the docID for this posting, return its list of positions
                    found = True
                    tokenPosInDoc.append(posting[1]) #get its positions in this document
                    tfidfScore = posting[2] #get its current tfidf
                    break
            if found:
                break

    indexPos = [0] * len(tokenPosInDoc) #indexPos is the current index you are looking at for that token in this document
    count = 0 #count is the number of times you found sequential words in the order of which they were entered
    lastIndex = len(indexPos) - 1 #save the last index

    minIndex = getMinIndex(indexPos, tokenPosInDoc) #gets the first index where this term appears in this document
    while(True):

        #if the minIndex is not the last index AND the current and the current token's position - the next token's position has a difference of only one (then they follow each other)
        if lastIndex != minIndex and tokenPosInDoc[minIndex][indexPos[minIndex]] - tokenPosInDoc[minIndex + 1][indexPos[minIndex + 1]] == 1:
            count += 1
            indexPos[minIndex] += 1 #incriment the next minimum index of this term
            if(len(tokenPosInDoc[minIndex]) <= indexPos[minIndex]): #if it reached the end of this list, break cuz the query n-gram is not not relavent
                break
                
            minIndex += 1 #make the minIndex one more as the next term is shown to follow this one
        else: #if the index after it is not the next word in the query and it is not the last index, then incriment the current term and compute the next index
            indexPos[minIndex] += 1 #incriment the next minimum index of this term
            if(len(tokenPosInDoc[minIndex]) <= indexPos[minIndex]): #if it reached the end of this list, break cuz the query n-gram is not not relavent
                break
            minIndex = getMinIndex(indexPos, tokenPosInDoc)

    return count

        

def getMinIndex(indexPos, tokenPosInDoc):
    i = 0
    minIndex = 0
    curMinVal = 0
    while i < len(indexPos):
        if tokenPosInDoc[i][indexPos[i]] < curMinVal:
            curMinVal = tokenPosInDoc[i][indexPos[i]] 
            minIndex = i
        i += 1
    
    return minIndex


def generate_boolean_or_search_result(boolean_query_list, intersection_docIDs):
    
    try:
        posting_union = []
        
        for index_obj in boolean_query_list: #index_obj = {'shindler': [[1948, [1065], 11.18], [3962, [133], 11.18], [6197, [37], 33.53], [7084, [82], 11.18], [7355, [92], 11.18], [8487, [20], 33.53], [9391, [32], 11.18], [11356, [31], 33.53], [17341, [58], 11.18]]}
            token = list(index_obj.keys())[0] # token = 'shindler'
            
            for posting in index_obj[token]:
                docID = posting[0]
                if docID not in intersection_docIDs:
                    posting_union.append([posting[0], posting[2]])
                    intersection_docIDs.add(docID)

        postings_union_ranked = sorted(posting_union, key=lambda x: x[1], reverse=True)
        
        
        return postings_union_ranked[0:50]

    except Exception as e:
        print('ERROR in generate_boolean_or_search_result: ', str(e))
        return []


def links_search_result(search_result, docId_to_urls):
    links = []
    for result in search_result:
        docID = result[0] # docID = result[0], result[1] = tfidf
        
        if str(docID) in docId_to_urls: #docId_to_urls = {docID: 'url'}
            links.append(docId_to_urls[str(docID)])
        else:
            print("THIS SHOULD NOT HAPPEN", result)
    return links



def launch_milestone_2(query, index_of_index, docId_to_urls, full_index):

    try:

        startTime = time.time()
        query_tokens = handle_stopwords(tokenizer(query))
        # ^^ tokenize query and then return a list of tokens after handling the stopwords
        boolean_query_list = []# term_being_searched: [post] => [{'decemb': [[4, [1826, 1917], 2]]}, {'deng': [[4, [222], 1]]}, {'depart': [[4, [2687], 1], [9, [108], 1], [12, [84], 1]]}]

        for token in query_tokens: # token is one of the words found in the query 
            if token in index_of_index:
                pos = index_of_index[token] # pos = is the position for this post in the full_indexs
                full_index.seek(pos)
                index_obj_txt = read_large_line(full_index) # index_obj_txt = "{'decemb': [[4, [1826, 1917], 2]]}"

                index_obj_json = json.loads(index_obj_txt) # index_obj_json = {'decemb': [[4, [1826, 1917], 2]]}
                
                boolean_query_list.append(index_obj_json) #push the index object to query list
            #if not in index_of_index, then we never countered that word, no result with that token. search result wont include this current token. but there might be result for the remaining tokens

        boolean_query_list_sorted = sorted(
            boolean_query_list, key=lambda x: len(list(x.values())[0])) #sorting based on docIDs, increasing order
        # [{'decemb': [[4, [1826, 1917], 2]]}, {'deng': [[4, [222], 1]]}, {'depart': [[4, [2687], 1], [9, [108], 1], [12, [84], 1]]}]

        boolean_and_result_docIDs_tfidf = boolean_and_search(boolean_query_list_sorted) #does boolean AND search, returns list of list => [[docID, tf-idf]]

        nGrams_search_result, intesection_docID = nGram_result(boolean_and_result_docIDs_tfidf, boolean_query_list) #nGrams_search_result = [[docID, tf-idf]], intesection_docID = set of docIDs found in boolean AND to avoid duplications in the boolean OR search

        #if there are not enough results, generate more search results using boolean OR
        or_boolean_search_result = generate_boolean_or_search_result(boolean_query_list_sorted, intesection_docID) if len(nGrams_search_result) < 50 else []
        # or_boolean_search_result = [[docID, tf-idf]]

        search_result = nGrams_search_result + or_boolean_search_result # search_result is combination of the result we got from nGrams and or_boolean_search_result
        endTime = time.time()
        finalTimeMS = (endTime - startTime) * 100

        print("Query time in mili-seconds: ", finalTimeMS)

        return links_search_result(search_result, docId_to_urls)
    

    except Exception as e:
        print('ERROR launch_milestone_2: ', str(e))
        return []

