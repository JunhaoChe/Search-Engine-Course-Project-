import glob
from indexer import tokenizer
from indexer import stopwords
from store_load import *
from compute_update import *
from config import *
opt_offset = {}
opt_index = None
urls_file = None
idf = {}
contents_file = None

def prepare():
    global opt_index
    global idf
    global urls_file
    global opt_offset
    global contents_file

    # Open all files and store file objects in a list
    with open(OPT_INDEX_OFFSET, 'r', encoding = 'utf-8') as o_file:
        opt_offset = json.load(o_file)
    with open(IDF, 'r', encoding = 'utf-8') as i_file:
        idf = json.load(i_file)
    with open(DOCUMENTS_CONTENTS, 'r') as c_file:
        contents_file = json.load(c_file)
    urls_file = open(URLS, 'r')
    opt_index = open(OPT_INDEX, 'r')


def close_all():
    global opt_index
    global urls_file
    if urls_file:
        urls_file.close()
    if opt_index:
        opt_index.close()

def query(input):
    terms = tokenizer(input)
    terms = [term for term in terms if term not in stopwords.words('english') and term in opt_offset]
    # find 5 highest idf tokens if exceed.
    if len(terms) > 5:
        terms = sorted(terms, key=lambda t: idf.get(t, 0), reverse=True)[:5]
    data_list = {term: load_token_data(opt_index, opt_offset[term]) for term in terms}
    top5 = get_top5(data_list, terms)
    print("-----Top 5 urls-----")
    urls = []
    for doc in top5:
        urls.append(read_url(urls_file, int(doc[0])))
    return urls

def get_top5(data_list, terms):
    if data_list is None or terms is None or len(terms) == 0:
        return []
    # if single token
    if len(data_list) == 1:
        doc_score = {}
        for data in data_list.values():
            for id_score in data:
                doc_score[id_score[0]] = id_score[1]
        top5 = sorted(doc_score.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Sinle token")
    elif len(data_list) > 1:
        doc_scores = defaultdict(float)
        for term in terms:
            for doc_id, td_idf in data_list[term]:
            # TF-IDF calculation: Term Frequency * Inverse Document Frequency
                doc_scores[doc_id] += td_idf

        # Sort documents by TF-IDF score in descending order
        sorted_docs_by_tf_idf = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Limit the number of documents to consider for cosine similarity calculation (e.g., top 20)
        top_docs_to_consider = sorted_docs_by_tf_idf[:20]

        # Calculate the cosine similarity for the top documents
        ranked_docs = []
        for doc_id, _ in top_docs_to_consider:
            similarity = compare_with_query(contents_file[doc_id], ' '.join(terms))
            ranked_docs.append((doc_id, similarity))
        
        top5 = sorted(ranked_docs, key=lambda x: x[1], reverse=True)[:5]
        print("Multiple tokens")
    print("top5: ", top5)
    return top5

# def query_with_word_positions(input): #extra credit for positions
#     doc_score = {}
#     terms = tokenizer(input)
#     terms = [term for term in terms if term not in stopwords.words('english')]
#     data_list = []

#     for term in terms:
#         first = term[0]
#         index = INDEX_MAP[first]
#         offset = offset_map[index].get(term, None)
#         if offset is not None:
#             file = file_handles[index]
#             data = load_token_data(file, offset)
#             data_list.append(data)

#     docs = get_common_doc_s_values(data_list)

#     for doc, scores in docs.items():
#         if len(scores) == len(terms):
#             position_bonus = 0
#             for term_data in data_list:
#                 if doc in term_data:
#                     positions = term_data[doc]['positions']
#                     if positions:
#                         position_bonus += max(positions) - min(positions)
#             total = sum(scores) - position_bonus  # Rewarding closer words
#             doc_score[doc] = total

#     # Sorting results by adjusted score
#     top5 = sorted(doc_score.items(), key=lambda x: x[1], reverse=True)[:5]

#     urls = []
#     for doc in top5:
#         urls.append(read_url(urls_file, int(doc[0])))
#     return urls

def run():
    prepare()
    while True:
        user_input = input("(To quit: ~q)Search query: ")
        if user_input == "~q":
            break
        start = time.time()
        urls = query(user_input)  # Using enhanced search function
        for url in urls:
            print(url)
        stop = time.time()
        print("Time: ", stop - start)
    close_all()

if __name__ == "__main__":
    run()