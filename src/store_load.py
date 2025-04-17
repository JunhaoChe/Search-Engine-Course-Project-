import json
import os
import re
import time
import ast
import sys
import subprocess
import traceback
from collections import defaultdict
from bs4 import BeautifulSoup
from config import INDEX_DIR, OFFSET_DIR, DATA_DIR, URLS, DOCUMENTS_CONTENTS, OPT_INDEX_DIR

try:
    import nltk
except ModuleNotFoundError:
    print("NLTK not found. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
    print("NLTK installed successfully.")

try:
    nltk.data.find('corpora/stopwords.zip')  # Check if stopwords are available
except LookupError:
    print("Stopwords dataset not found. Downloading now...")
    nltk.download('stopwords')
from nltk.corpus import stopwords

try:
    from bitarray import bitarray
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'bitarray'])


if not os.path.exists(OFFSET_DIR):
    os.makedirs(OFFSET_DIR)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def store_urls(sorted_urls, filename, fixed_length=300):
    try:
        with open(filename, 'wb') as f:
            for url in sorted_urls:
                if len(url) < fixed_length-1:
                    padded_url = url.ljust(fixed_length, '\0')  # Pad with null characters
                    f.write(padded_url.encode('utf-8'))
                else:
                    print("url too long, fixed_length should be bigger")
                    print(url)
                    return False
    except IOError:
        print("Could not write to file {}".format(filename))

def read_url(file, index, fixed_length=300):
    if index > 0:
        index = index - 1
    if file:
        file.seek(index * fixed_length)  # Jump to the position
        url = file.read(fixed_length).strip('\0')  # Read and clean padding
        return url
    else:
        print("Could not open file\n")


# store in json files using file.tell() to record positions
def store_token_offset(infile, outfile):
    print("Reading from {}".format(infile))
    print("Writing to {}".format(outfile))
    token_offsets = {}

    # Create the data file
    with open(f"{DATA_DIR}{outfile}.txt", 'w', encoding='utf-8') as data_file:
        # Read the JSON index
        with open(infile, 'r', encoding='utf-8') as f:
            try:
                index_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: {infile} is not a valid JSON file or is empty")
                return

        for token, postings in index_data.items():

            position = data_file.tell()
            token_offsets[token] = position

            # postings_str = json.dumps(postings) + "\n"
            postings_list = []
            for item in postings:
                for id,attribute in item.items():
                    postings_list.append((id, attribute['s']))
            data_file.write(str(postings_list)+'\n')

    with open(f"{OFFSET_DIR}{outfile}_offsets.json", 'w', encoding='utf-8') as offset_file:
        json.dump(token_offsets, offset_file)


# read token's info from a txt file using file.seek()
def load_token_data(file, offset):
    if file:
        file.seek(offset)
        line = file.readline().strip()
        return ast.literal_eval(line)
    else:
        raise FileNotFoundError(f"file {file} not found")

def load_doc(file, offset):
    if file:
        file.seek(offset)
        return file.readline().strip()
    else:
        raise FileNotFoundError(f"file {file} not found")

def create_store_TDM(inverted_indexes_dir, total, store_dir):
    tdm = defaultdict(lambda: bitarray(total))  # Default to a bitarray of size 'total'

    # Process inverted indexes
    for filename in os.listdir(inverted_indexes_dir):
        if filename.endswith(".json"):
            filePath = os.path.join(inverted_indexes_dir, filename)
            with open(filePath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for token, postings in data.items():
                    if token not in tdm:
                        tdm[token] = bitarray(total)
                        tdm[token].setall(0)  # Initialize with zeros

                    for posting in postings:
                        for doc_id, attributes in posting.items():
                            doc_id = int(doc_id)
                            if 0 <= doc_id < total:
                                tdm[token][doc_id] = 1

    # Ensure store directory exists
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)

    # Open files for writing
    token_file_path = os.path.join(store_dir, "tokens.txt")
    bitarray_file_path = os.path.join(store_dir, "bitarrays.bin")

    with open(token_file_path, 'w', encoding='utf-8') as token_file, open(bitarray_file_path, 'wb') as bitarray_file:
        for token, bitarr in tdm.items():
            # Write token to token file (string format)
            token_file.write(token + '\n')

            # Write bitarray as binary to the bitarray file
            bitarray_file.write(bitarr.tobytes())  # Write the bitarray as bytes

def load_TDM(token_file_path, bitarray_file_path, total):
    tdm = {}

    token_file =  open(token_file_path, 'r', encoding='utf-8')
    bitarray_file = open(bitarray_file_path, 'rb')
    while True:
        token = token_file.readline().strip()  # Read token
        if not token:
            break  # Stop if no more tokens

        bitarr_bytes = bitarray_file.read(total // 8 + (1 if total % 8 else 0))  # Read the bitarray in binary
        if not bitarr_bytes:
            break
        bitarr = bitarray()
        bitarr.frombytes(bitarr_bytes)  # Convert bytes back to bitarray

        tdm[token] = bitarr
    token_file.close()
    bitarray_file.close()

    return tdm
def get_intersection(bitarrays):
    intersection = bitarrays[0]
    for bit in bitarrays[1:]:
        intersection = bit&intersection
    # Find the positions of intersected document IDs (where bit is '1')
    intersected_doc_ids = [index for index, bit in enumerate(intersection) if bit == 1]
    print("Intersected Document IDs:", intersected_doc_ids)
    return intersected_doc_ids

"""
    html reader and process the contend(stemming)
"""
def content_process(content):
    from indexer import tokenizer
    if not content:
        return None
    soup = BeautifulSoup(content, 'html.parser')
    for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'noscript', 'link']):
        tag.decompose()
    text = ' '.join(soup.stripped_strings)
    all_tokens = tokenizer(text)
    text = ' '.join(all_tokens)
    text = text
    return text

"""
    process the html contents and store them in a file for further processing
"""
def store_doc(root_directory, store_path, urls_path = URLS):
    try:
        url_file = open(urls_path, 'r')
        doc_id = 1
        docs = {}
        for subdomain in os.listdir(root_directory):
            subdomain_path = os.path.join(root_directory, subdomain)
            if os.path.isdir(subdomain_path):
                print(f"Processing Subdomain: {subdomain}")
                for file in os.listdir(subdomain_path):
                    file_path = os.path.join(subdomain_path, file)
                    if file.endswith(".json"):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            exist_url = read_url(url_file, doc_id)
                            if not exist_url or len(exist_url) == 0:
                                break
                            data = json.load(f)
                            url = data.get('url', '')
                            # check if the url was processed during indexing
                            if exist_url == url:
                                content = data.get('content', '')
                                text = content_process(content)
                                # store the content and record the offset for faster retrieval
                                # url_offset[doc_id] = store_file.tell()
                                # store_file.write(text)
                                docs[doc_id] = text
                                doc_id = doc_id + 1

        url_file.close()
        with open(store_path, 'w', encoding='utf-8') as f:
            json.dump(docs, f,indent=1)


    except IOError as e:
        print("Store_doc error: " + e)
        traceback.print_exc()



def append_key_value_to_json(file_path, new_data):
    try:
        # Step 1: Read existing data
        with open(file_path, "r") as file:
            data = json.load(file)

        # Step 2: Ensure it's a dictionary
        if isinstance(data, dict):
            data.update(new_data)  # Append new key-value pairs
        else:
            raise ValueError("JSON file must contain a dictionary at the top level.")

    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is empty, create a new dictionary
        data = new_data

    # Step 3: Write updated data back to file
    with open(file_path, "w") as file:
        json.dump(data, file, indent=1)
# call it after finish recording all the processed documents
def process_and_rank_documents(file_path, store_path, contents_file, max_doc):
    print(file_path)
    from compute_update import rank_documents
    # open for reading document content.
    with open(contents_file, 'r', encoding='utf-8') as f:
        id_contents = json.load(f)

    new_data = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as f :
        index_data = json.load(f)
        for token, postings in index_data.items():
            if token in stopwords.words('english') or len(token) < 2:
                continue
            documents = []
            ids = []
            for posting in postings:
                for doc_id, att in posting.items():
                    content = id_contents[doc_id]
                    documents.append((doc_id, content))
                    ids.append(doc_id)
            ranked_docs = ids
            if len(ids) >= max_doc:
                ranked_docs = rank_documents(documents)
            top_docs = ranked_docs[:max_doc]
            sorted_top_docs = sorted(top_docs)
            new_data[token] = sorted_top_docs
            if len(new_data) >= 2000:
                append_key_value_to_json(store_path,new_data)
                new_data.clear()
    if len(new_data) > 0:
        append_key_value_to_json(store_path, new_data)

"""
    Finding the best documents for each token based on the cosine similarity
    By default, finding the top 20
"""
def optimize_inverted_index(max_doc=20, directory=INDEX_DIR):
    store_dir = OPT_INDEX_DIR
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        store_path = os.path.join(store_dir, file)
        process_and_rank_documents(file_path, store_path, DOCUMENTS_CONTENTS, max_doc)

"""
    retrieve "tf_idf" value
"""
def get_tf_idf(data, token, doc_id):
    doc_id = str(doc_id)
    if token in data:
        for entry in data[token]:
            if doc_id in entry:
                return entry[doc_id]["s"]
    return None

"""
    Since the optimized inverted indexes do not contain the tf_idf,
    lookup the tf_idf in the inverted indexes and place them into the optimized inverted indexes
"""
def optimize_inverted_index_plus_tf_idf(optimize_index_directory, original_index_directory):
    for file in os.listdir(optimize_index_directory):
        original_file_path = os.path.join(original_index_directory, file)
        optimize_file_path = os.path.join(optimize_index_directory, file)
        new_data = defaultdict(list)
        with open(original_file_path, 'r', encoding='utf-8') as ori_f , open(optimize_file_path, 'r', encoding='utf-8') as opt_f:
            ori_data = json.load(ori_f)
            opt_data = json.load(opt_f)
            for token, postings in opt_data.items():
                for doc_id in postings:
                    tf_idf = get_tf_idf(ori_data, token, doc_id)
                    new_data[token].append((doc_id, tf_idf))

        with open(optimize_file_path, 'w', encoding='utf-8') as opt_f:
            json.dump(new_data, opt_f, indent=4)

"""
    store the optimized inverted indexes and the positions of token info in files.
"""
def store_and_record_offset_for_optimized_index(optimize_index_directory,store_path, offsets_path):
    store_file = open(store_path, 'w', encoding='utf-8')
    offsets = {}
    for file in os.listdir(optimize_index_directory):
        optimize_file_path = os.path.join(optimize_index_directory, file)
        with open(optimize_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for token, postings in data.items():
                offsets[token] = store_file.tell()
                store_file.write(str(postings)+'\n')
    with open(offsets_path, 'w', encoding='utf-8') as f:
        json.dump(offsets, f, indent=1)
    store_file.close()

