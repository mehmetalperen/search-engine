import re
import json
import os
import os.path
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import validators
from urllib import robotparser
import shelve
from urllib.parse import urljoin
import hashlib
''' the commented out libraries don't import correctly '''
# from porter2stemmer import Porter2Stemmer
# from simhash import Simhash
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer  # to stem
from stop_words import get_stop_words
import shelve
from simhash import Simhash


file_count = 0
index_split_counter = 0
docID = 0
inverted_index = {}
docID_urls = {}
index_of_index = {}
simhash_scores = [] #list of simhash object for all the documents. to detect the pages with duplicate content

# class Posting:
#     def __init__(self, docID, token_locs, tfidf):
#         self.docId = docID
#         self.token_locs = token_locs
#         self.tfidf = tfidf       # word frequency

# instead, lets try to define a list



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
            if ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.-'":
                if ch in ".-'":
                    continue
                cur_word += ch.lower()  # convert that ch to lower case and add it to the cur_word
            elif len(cur_word) > 0:
                tokens.append(stemmer.stem(cur_word))
                cur_word = ""
        if len(cur_word) > 1:
            tokens.append(stemmer.stem(cur_word))
        return tokens  
    except Exception as e:
        print(f"Error tokenizer: {str(e)}")
        return []
    # stemmed_tokens = [stemmer.stem(token) for token in tokens]



def get_file_text_content(file_path):
    '''
    get_file_text_content gets the html content from a json file and
    extracts its text. It also gets each tags and counts the important/
    words in a a dictionary with {term-str: total weight-int}
    Returns: (text_content-str, bold_word_counter-dict)
        text_content: the text content of a json filecounter of the total
        bold_word_counter: dict of {term-str: total weight-int}
    '''

    try:
        with open(file_path, 'r') as f:
            data = json.load(f) #load json file
            html_content = data['content']
            soup = BeautifulSoup(html_content, 'html.parser')
            text_content = soup.get_text() #get text content
            #for each tag, get its text content
            ##if it is important, store its weight and increment
            ##each word in the tag according to that weight in bold_word_counter
            bold_word_counter = {} # key = the word found in important tags. value is how many times we see that word
            
            for tag in soup.findAll(): # looping all the tags
                tag_name = tag.name # current tag name
                tag_text_content = tag.get_text() #content in the tag => <h1> example </h1> tag_text_content is "example"
                increment_by = 0 # will have different weight for different tags.
                if tag_name == 'h1':
                    increment_by = 4
                elif tag_name == 'h2':
                    increment_by = 3
                elif tag_name == 'h3':
                    increment_by = 2
                elif tag_name == 'h4' or tag_name == 'h5' or tag_name == 'h6' or tag_name == 'stong':
                    increment_by = 1
                
                if increment_by != 0: # if increment_by is zero, then we did not find an important tag
                    for word in tokenizer(tag_text_content):
                        if word in bold_word_counter:
                            bold_word_counter[word] += increment_by
                        else:
                            bold_word_counter[word] = increment_by
                        
            return (text_content, bold_word_counter) if text_content else (None, bold_word_counter)
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return (None, {})



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


def map_docID_url(file_path, docID):
    '''
    map_docID_url maps docID to its URL in the dictionary in file_path
    Returns: null
    '''
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            url = data['url']
            docID_urls[docID] = url
    except Exception as e:
        print(f"Error Mappign DocID to URL {file_path}, {docID} : {str(e)}")


def get_file_paths(folder_path):
    '''
    get_file_paths gives a list of paths of all the json files
    Returns: filepath of all json files (list)
    '''
    paths = []
    for dirpath, dirnames, filenames in os.walk(folder_path): #goes in to DEV folder and gets the filename found in the dev folder => dirpath = directory path, dirnames = directory names, filenames = filenames
        for filename in filenames: # filenames is a list of all the files found in the DEV folder
            if filename.endswith('.json'): # get the json files
                file_path = os.path.join(dirpath, filename) # join directory path and filename to create the path of the json file
                paths.append(file_path) # add the path to the list that we will be return
    return paths


def write_to_file(thisFile, newDict):
    '''
    write_to_file writes newDict to thisFile with a newline
    between each dictionary key
    Returns: null
    '''
    for key in newDict:
        tempDict = {key: newDict[key]}
        json.dump(tempDict, thisFile)
        thisFile.write('\n')


def generate_inverted_index(token_locs, docID, strong_word_count):
    '''
    generate_inverted_index fills/writes out the inverted_index. Write the
    inverted_index to a file if the number of documents since last write exceeds 5000.
    - token_locs (dict) that has a word (str) as a key and a value as a list of positions of that word in the document
    - docID is the document's ID
    - strong_word_count (dic of strong words and count)
    Returns: null, writes to global variable inverted_index (dict)
    '''

    global index_split_counter
    global file_count
    index_split_counter += 1

    #if this is true, write inverted_index to index#.txt
    #and reset indexSplitCounter to 0
    if index_split_counter > 5000:
        fileName = "index" + str(file_count) + ".txt"
        if os.path.exists(fileName):
            os.remove(fileName)
        with open(fileName, "w") as thisFile:
            res = sorted(inverted_index.items())
            newDict = dict(res)
            write_to_file(thisFile, newDict)  

        inverted_index.clear()
        index_split_counter = 0
        file_count += 1
    try:
        for token in token_locs:
            tfidf = 0
            # if its found in strong_word_count, incriment make its
            # tfidf equal to the number of tokens + strong_word_count[token]
            # then decriment the strong_word_count for that token
            if token in strong_word_count and strong_word_count[token] > 0:
                tfidf = len(token_locs[token]) + strong_word_count[token]
                strong_word_count[token] -= 1
            else:
                tfidf = len(token_locs[token]) #if no strong word tfidf = the # of occurrances of this token
                

            # post = Posting(docID, token_locs[token], tfidf)
            post = [docID, token_locs[token], tfidf]
            #write out each token to the inverted_index
            if token in inverted_index:
                inverted_index[token].append(post)
            else:
                inverted_index[token] = [post]
    except Exception as e:
        print(f"Error Generating Inverted Index {docID} : {str(e)}")


def write_remaining_index():
    '''
    write_remaining_index writes out inverted_index to a new file (will be thefinal partial index).
    Returns: null, writes out the global variable inverted_index (dict)
    '''

    global index_split_counter
    global file_count
    index_split_counter += 1

    filename = "index" + str(file_count) + ".txt"
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, "w") as cur_file:
        res = sorted(inverted_index.items())
        newDict = dict(res)
        
        write_to_file(cur_file, newDict)  # essentially json.dump(newDict, thisFile) with new lines

    inverted_index.clear()
    index_split_counter = 0
    file_count += 1


def get_key(key):
    '''
    getKey gets the current key in the line of text representing a dictionary
    Returns: key (str)
    '''
    first_quote = key.find('"')
    second_quote = key.find('"', first_quote + 1)

    return "" if first_quote == -1 or second_quote == -1 else key[first_quote + 1: second_quote] # if there is no key, return empty string


def merge_step(temp_dict, dict2):
    '''
    merge_step merges two given dictionaries (in this case tokens) together from the passed dictionaries
    - technically, there should only be one key per dictionary (see merge_partial_indexes)
    '''
    for key in dict2:
        if key in temp_dict:
            i = 0
            k = 0
            dict2_list = dict2[key]
            # for every posting for this token, check if the posting's docID (posting[0])
            # is greater than the the current posting at dict2List[k]'s docID
            # if it is, add insert it in the dictionary (preserves docID ordering)
            for posting in temp_dict[key]:
                if (posting[0] > dict2_list[k][0]):
                    temp_dict[key].insert(i, dict2_list[k])
                    k = + 1
                    if k >= len(dict2_list):  # reached the end
                        break
                i += 1

            # if we havent reached the end of dict2List, append the remaining postings
            # to dictHolder
            while k < len(dict2_list):  
                temp_dict[key].append(dict2_list[k])
                k += 1

        else:
            temp_dict[key] = dict2[key]


def merge_partial_indexes():
    '''
    merge_partial_indexes merges the partial indexes into one file: full_index.txt
    Returns: null
    '''
    if os.path.isfile("full_index.txt"):
        os.remove("full_index.txt")
    full_index = open("full_index.txt", 'w')
    global file_count
    temp_count = 0
    files = []

    # open all files and write their file pointers to arrFiles
    while temp_count < file_count:  
        filename = "index" + str(temp_count) + ".txt"
        temp_holder = open(filename, "r")
        files.append(temp_holder)
        temp_count += 1
    nxt_min_indexes_txts = []
    nxt_min_indexes_dict = []
    
    temp_count = 0

    while temp_count < file_count:
        # tempStr is a list of dictionary entries represented by text
        tempStr = read_large_line(files[temp_count])
        
        if tempStr:
            nxt_min_indexes_txts.append(tempStr)

            # will be a list of ACTUAL dictionary entries
            nxt_min_indexes_dict.append(
                json.loads(nxt_min_indexes_txts[temp_count]))
        temp_count += 1

    while (True):
        # MINKEY: populates arrNextMinIndexesText and gets the minimum key out of them
        min_key = ""
        for x in nxt_min_indexes_txts:  # gets the first non-empty string and assign it to minKey
            if x != "":
                min_key = x
                break
        if min_key == "":  # means that all of them were empty strings, nothing else to read from all files
            break

        #gets the minimum key (reasoning is cuz we want to preserve alphabetical order of the keys)
        for x in nxt_min_indexes_txts:
            cur_key = get_key(x)
            if cur_key != "" and (cur_key < min_key):
                min_key = cur_key
        i = 0

        temp_dict = {}  # holder is the new dictionary which will hold only one token (current minimum)
        # Below loop essentially gets and merdges all of the files which have the minimum token (alphabetically speaking)
        while i < file_count:
            if min_key in nxt_min_indexes_dict[i]:
                merge_step(temp_dict, nxt_min_indexes_dict[i]) #merge
                nxt_min_indexes_txts[i] = read_large_line(files[i])  # read the next line in this file
                if (nxt_min_indexes_txts[i] != ""):
                    # update this to the next dict entry
                    nxt_min_indexes_dict[i] = json.loads(
                        nxt_min_indexes_txts[i])
            i += 1
        
        # write dictHolder to the full_index
        json.dump(temp_dict, full_index)
        full_index.write('\n')

    temp_count = 0
    while temp_count < file_count:  # close all files loop
        files[temp_count].close()
        temp_count += 1
    full_index.close()


def token_locator(tokens):
    '''
    token_locator takes in a list of tokens from a document and returns...
    Returns: token_locs, dict of {word(str) : list of positions of where the token appears in the document (list of ints)}
    '''
    token_locs = {}
    i = 0
    for token in tokens:
        if not token:
            continue
        if token in token_locs:
            token_locs[token].append(i)
        else:
            token_locs[token] = [i]
        i += 1

    return token_locs

# THIS FUNCTION IS NOT USED IN MILESTONE 3, don't need to review
def generate_report():
    '''generate_report generates our report for milestone 1. It will print the word and the list of all the documents that word seen and frequency of that word in that doc. for example, "random_word": [(1, 0.24) (99, 0.0029) ... ] where every tupple is (docID, frequency)'''
    try:
        filename = 'REPORT.txt'
        file2 = 'InvertedIndex.txt'

        if os.path.isfile(filename):
            os.remove(filename)

        if os.path.isfile(file2):
            os.remove(file2)

        file = open(filename, 'w')
        file.write("REPORT: \n")

        inverted_inxex_txt = open(file2, 'w')

        file.write('Number of indexed documents: ' + str(docID) + '.\n')

        file.write('Number of unique words: ' +
                   str(len(inverted_index)) + '.\n')

        for token in inverted_index:
            inverted_inxex_txt.write(token + ": [ \n")
            new_line_count = 0
            for post in inverted_index[token]:
                # InvertedIndexTXT.write("(" + str(post.docId) +
                #                        ", " + str(post.token_locs) + ', ' + str(post.tfidf) + ') ')
                inverted_inxex_txt.write("(" + str(post[0]) +
                                       ", " + str(post[1]) + ', ' + str(post[2]) + ') ')
                new_line_count += 1
                if new_line_count >= 10:
                    inverted_inxex_txt.write('\n')
                    new_line_count = 0

            inverted_inxex_txt.write('] \n------------------------------\n')


        file_size = os.path.getsize(file2)
        file.write('Size of the inverted index: ' +
                   str(file_size // 1024) + ' KB.\n')
        file.close()
        inverted_inxex_txt.close()
        print('DONE')
    except Exception as e:
        print(
            f"Error Generating Report: {str(e)}")


def is_duplicate_content(text_content):
    '''
    is_duplicate_content: if we add a page with similar content, return true
    - uses Simhash to detect duplicates with a threshold of 12
    '''
    global simhash_scores #simhash objects (list of previous hashes)
    
    finger_print = Simhash(text_content)
    for other_fingerprint in simhash_scores:  # loop through each one        
        similarity = finger_print.distance(other_fingerprint) # see if they are similar
        if similarity <= 12: # 0 = 100 % same, 64 = 0 % same ||||| ran 17, 30, 12, 15, 13 (13 gave 11k) (never finished 12 thinking it got into a trap. it may or may not be true)
            return True #if they are similar, return True

    simhash_scores.append(finger_print) #if they are not similar, add it to simhash_scores
    return False

def create_index_of_index():
    '''
    create_index_of_index: creates the index of index
    - gets the beginning position of each line in full_index.txt (which is the posting list for a token)
      and stores it in a dictionary with the token string as a key
    Returns: null but writes out the index of index to index_of_index.txt
    '''
    full_index = open("full_index.txt", 'r')
    while True:
        pos = full_index.tell()
        curLine = read_large_line(full_index)
        if not curLine:
            break  # need to break here if the line is empty
        tempDict = json.loads(curLine)
        for token in tempDict:
            if token in index_of_index:
                print("error, index shouldn't already exist")
            else:
                index_of_index[token] = pos

    if os.path.isfile("index_of_index.txt"):
        os.remove("index_of_index.txt")
    json.dump(index_of_index, open("index_of_index.txt", "w"))


def launch_milestone_1():
    '''
    Essentially executes mileStone1.py which creates and merdges the partial indexes
    as well as the index of indexes of the full index.
    '''
    folder_path = '/home/mnadi/121/final-search-engine/search-engine/DEV'
    
    #store duplicate pages in case we want to see them in duplicate_pages.txt
    if os.path.isfile("duplicate_pages.txt"):
        os.remove("duplicate_pages.txt")

    paths = get_file_paths(folder_path)  # list of paths to all the files
    
    duplicate_pages_txt = open('duplicate_pages.txt', 'w')
    global docID

    #go through each file path and add its text content to an inverted_index
    for path in paths:
        text_content, bold_word_counter = get_file_text_content(path)
        if not text_content:  # skip if no text content
            continue
        if is_duplicate_content(text_content): # if we have a page with really similar content of this current document, we will not add it to the index.
            with open(path, 'r') as f: # we wanna have a record of the pages that we did not add to our index
                data = json.load(f)
                url = data['url']
                duplicate_pages_txt.write(url)
                duplicate_pages_txt.write('\n')
            continue
        
        docID += 1 
        map_docID_url(path, docID) # assign docID to its proper URL // {docID : url}
        tokens = tokenizer(text_content)  # tokenize the text content

        token_locs = token_locator(tokens)  # get a list of token positions

        generate_inverted_index(token_locs, docID, bold_word_counter)

    
    if os.path.isfile("total_doc_count.txt"): #total_doc_count.txt will help us calculate tfidf later
        os.remove("total_doc_count.txt")
    
    
    total_doc_count = open("total_doc_count.txt", 'w')
    total_doc_count.write(str(docID))
    total_doc_count.close()
    write_remaining_index()  # write the remaining data out in inverted_index to another partial index
    merge_partial_indexes()  # merges the partial indexes
    create_index_of_index()  # creates index_of_index (dictionary in a file)
    duplicate_pages_txt.close()
    if os.path.isfile("docID_urls.txt"):
        os.remove("docID_urls.txt")
    json.dump(docID_urls, open("docID_urls.txt", "w"))




if __name__ == '__main__':
    print("Running...")
    launch_milestone_1()
    print('Done')