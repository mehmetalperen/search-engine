
import json
import os

import math

def create_index_of_index():
    full_index = open("full_index_tf_idf.txt", 'r') # create new full index with tf-idf scores
    index_of_index = {} # positions of full index will change, need to create new index_of_index
    
    while True:
        pos = full_index.tell() #current position in full_index
        curLine = read_large_line(full_index) # reading line

        if not curLine: # end of line
            break  # end of line, EXIT reading
        tempDict = json.loads(curLine)
        for token in tempDict:
            if token in index_of_index:
                print("error, index shouldn't already exist")
            else:
                index_of_index[token] = pos

    if os.path.isfile("index_of_index_tf_idf.txt"):
        os.remove("index_of_index_tf_idf.txt")
    json.dump(index_of_index, open("index_of_index_tf_idf.txt", "w"))
    

def calculate_tf_idf(index_obj, key, total_doc_count):
    '''
    formula = (1 + log(tf)) * log(N / df)
    tf = len(posting[1]) AKA term_frequency 
    N = total_doc_count
    df = len(index_obj[key]) AKA document_frequency

    '''
    try:
        N = total_doc_count
        df = len(index_obj[key])
        for posting in index_obj[key]: # posting = [28695, [30088, 70088], 2] IN = [[28695, [30088, 70088], 2], [31095, [152875], 1], [32761, [30088, 70088], 2], [40645, [14678], 1], [41010, [355455], 1], [55365, [5436462, 6634496, 9478486], 3]]
            #posting[0] = docID
            #posting[1] = positions "key" found in docID 
            #len(posting[1]) = term_frequency
            # posting[2] = where we will put td-idf
            tf = posting[2]

            tf_idf = round((1 + math.log(tf, 2)) * math.log((N / df), 2), 2)
            # print('before: ', posting[2])
            posting[2] = tf_idf
            # print('after: ', posting[2])
            
        
        return index_obj
    except Exception as e:
        print("ERROR in calculate_tf_idf: ", str(e))
        
        
def generate_full_index_tf_idf():
    '''generates a txt file of a full_index with actaul tf-idf scores'''
    
    
    try:
        full_index = open('full_index.txt', 'r') #open full index with NO tf-idf scores
        total_doc_count = int(open('total_doc_count.txt', 'r').readline()) # get total doc amount for our tf-idf formula




        if os.path.isfile("full_index_tf_idf.txt"): #create new full index with tf-idf scores
            os.remove("full_index_tf_idf.txt")
        full_index_tf_idf = open('full_index_tf_idf.txt', 'w')

        
        while True:

            index_txt= read_large_line(full_index)
            if not index_txt: #if line is empty, end is reached, exit
                break
            index_obj = json.loads(index_txt) #{"000000000000003518": [[33278, [2697], 1]]}
            key = list(index_obj.keys()) 
            key = key[0] #key = '000000000000003518'
            index_obj_tf_idf = calculate_tf_idf(index_obj, key, total_doc_count)

            json.dump(index_obj_tf_idf, full_index_tf_idf)
            full_index_tf_idf.write('\n')
            # testing_break += 1 #remove this 
        full_index_tf_idf.close()
        full_index.close()
    except Exception as e:
        print("ERROR in generate_full_index_tf_idf: ", str(e))

def read_large_line(file):
    chunk_size = 4096  # python line buffer size
    line = ''

    while True:
        chunk = file.readline(chunk_size)
        line += chunk

        if len(chunk) < chunk_size or '\n' in chunk:
            break

    return line





if __name__ == '__main__':
    print('running...')
    generate_full_index_tf_idf()
    create_index_of_index()
    print('done')