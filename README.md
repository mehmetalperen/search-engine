# search_engine
The file should include mileStone1.py, mileStone2.py, calculateTFIDF.py, app.py, /DEV and /templates
- /DEV contains the json files with the html pages
- /templates contains index.html and results.html

## Installation:
The following packages are commonly not installed on machines and 
may need to be installed on your local system in order to run the file (though there may be more):

- ntlk: Porter Stemmer
- bs4: BeautifulSoup
- psutil
- simhash
- flask
- stop_words

Also, python3 should be installed on your machine.

## Usage:
Step 1: mileStone1.py : creates the index of index and partial indexes and merges the partial indexes

Step 2:	calculateTFIDF.py : creates the tf-idf scores for each document in the full index (outputs as full_index_tf_idf.txt). Also updates the index of index (outputs as index_of_index_tf_idf.txt)

Step 3:	 app.py : runs the search using search from mileStone2.py
	 	  which renders the search input to index.html. Once a search is
         	  performed, it renders the result to results.html and waits for
         	  another query input. Perform a query by typing in the text box on the GUI
		  and returning.
