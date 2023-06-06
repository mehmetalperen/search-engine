from flask import Flask, render_template, request
import json
import mileStone1
import mileStone2
from mileStone2 import launch_milestone_2

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def perform_search():
    # Get the query from the request arguments
    query = request.args.get('query')

    # Perform the actual search and get all results
    all_results = perform_actual_search(query)

    # Separate the returned results so that 10 urls are returned
    # per page.

    page = request.args.get('page', 1, type=int)
    results_per_page = 10
    start_index = (page - 1) * results_per_page
    end_index = start_index + results_per_page
    results = all_results[start_index:end_index]

    # Render the results template with the query and paginated results
    return render_template('results.html', query=query, results=results, page=page)

#implements the actual search engine to return relevent results to the HTML page
def perform_actual_search(query):
    
    index_of_index = json.load(open("index_of_index_tf_idf.txt"))
    docId_to_urls = json.load(open('docID_urls.txt'))
    full_index = open('full_index_tf_idf.txt', 'r')

    # Call the launch_milestone_2 function to perform the search
    queryURLs = launch_milestone_2(query, index_of_index, docId_to_urls, full_index)

    # If no results are found, add a dummy message
    if len(queryURLs) == 0:
        queryURLs.append("No Results")

    return queryURLs

if __name__ == '__main__':
    app.run(port=5001)