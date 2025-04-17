import os
import json
import traceback
from collections import defaultdict
from bs4 import BeautifulSoup
import hashlib
from store_load import *
import re
from compute_update import *
from config import DATASET, URLS, IDF
import subprocess
import sys
# Check if simhash is installed, if not, install it
try:
    from simhash import Simhash
except ImportError:
    print("Simhash is not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "simhash"])
    from simhash import Simhash  # Import after installing

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
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


TAG_WEIGHTS = {
    'title': 5,
    'h1': 3,
    'h2': 2,
    'h3': 1.5,
    'strong': 1.2,
    'other' : 1
}

inverted_index = {}
word_occurrence = {}
id_counter = 0
BATCH_SIZE = 3000
simhash_fingerprints = []
seen_checksums = set()
urls = []

def generate_checksum(content):
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def generate_simhash(tokens):
    return Simhash(tokens).value

def find_near_duplicates(new_fingerprint, threshold=5):
    for fingerprint in simhash_fingerprints:
        hamming_distance = bin(new_fingerprint ^ fingerprint).count('1')
        if hamming_distance <= threshold:
            return True
    simhash_fingerprints.append(new_fingerprint)
    return False

def is_exact_similar(new_fingerprint):
    for fingerprint in simhash_fingerprints:
        hamming_distance = bin(new_fingerprint ^ fingerprint).count('1')
        if hamming_distance == 0:
            return True
    simhash_fingerprints.append(new_fingerprint)
    return False


def tokenizer(text):
    ps = PorterStemmer()
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    tokens = [ps.stem(tok) for tok in tokens]
    return tokens

def content_parser(content):
    if not content:
        return None
    soup = BeautifulSoup(content, 'html.parser')
    for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'noscript', 'link']):
        tag.decompose()
    text = ' '.join(soup.stripped_strings)
    all_tokens = tokenizer(text)
    simhash_fingerprint = generate_simhash(all_tokens)
    if find_near_duplicates(simhash_fingerprint):
        return {}
    tag_names = ['title', 'h1', 'h2', 'h3']
    bold_tags = ['b', 'strong']

    # Extract and tokenize text for each tag
    tag_sets = {tag: set(tokenizer(' '.join(t.get_text() for t in soup.find_all(tag)))) for tag in tag_names}
    tag_sets['strong'] = set(tokenizer(' '.join(t.get_text() for t in soup.find_all(bold_tags))))

    # Extract and process anchor words
    anchor_words = defaultdict(set)
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        anchor_text = tokenizer(a_tag.get_text())
        for word in anchor_text:
            anchor_words[word].add(href)  # Associate anchor words with target URLs

    # Initialize word info dictionary
    word_info = defaultdict(lambda: {'f': 0, 'w': 0.0, 'positions': [], 'anchors': []})
    for pos, token in enumerate(all_tokens):
        word_info[token]['f'] += 1
        word_info[token]['positions'].append(pos)
        max_weight = TAG_WEIGHTS['other']

        for tag, token_set in tag_sets.items():
            if token in token_set:
                max_weight = max(TAG_WEIGHTS[tag], max_weight)

        word_info[token]['w'] = max_weight
        # Add anchor links if the word appears as an anchor
        if token in anchor_words:
            word_info[token]['anchors'] = list(anchor_words[token])
    total_tokens = len(all_tokens)
    if total_tokens > 0:
        for token, attributes in word_info.items():
            attributes["f"] = (attributes["f"] / total_tokens) * attributes["w"]
    return word_info if word_info else {}

def update_inverted_index(word_info, id):
    global inverted_index
    for token, attributes in word_info.items():
        if token in inverted_index:
            inverted_index[token].append({str(id): {"f": attributes["f"], "w": attributes["w"], "positions": attributes["positions"]}})
        else:
            inverted_index[token] = [{str(id): {"f": attributes["f"], "w": attributes["w"], "positions": attributes["positions"]}}]

def get_partition_filename(token):
    """
    partitioned by 26 alphabet and 0-9
    """
    if not token:
        return "index_other.json"

    first_char = token[0].lower()
    if first_char.isalpha():
        return f"index_{first_char}.json"
    elif first_char.isdigit():
        return f"index_{first_char}.json"
    else:
        return "index_other.json"

# File creation. Note: do it before indexing
def create_files():
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
    try:
        empty = {}
        for i in range(ord('a'), ord('z')+1):
            filepath = f"{INDEX_DIR}/index_{chr(i)}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(empty, f)
        for i in range(ord('0'), ord('9')+1):
            filepath = f"{INDEX_DIR}/index_{chr(i)}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(empty, f)
    except IOError as e:
        print(f"File creation has failed: {e}")


def save_inverted_index_by_partitions(inverted_index, base_path=INDEX_DIR):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    partitioned = {}
    for token, postings in inverted_index.items():
        partition = get_partition_filename(token)
        if partition not in partitioned:
            partitioned[partition] = {}
        partitioned[partition][token] = postings
    for partition, new_data in partitioned.items():
        filepath = os.path.join(base_path, partition)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {}
        else:
            existing_data = {}

        for token, postings in new_data.items():
            if token in existing_data:
                existing_data[token].extend(postings)
            else:
                existing_data[token] = postings

        # Write the merged data back to the file.
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=4)
        except IOError as e:
            print(f"File updates have failed: {e}")


def read_json_files(root_directory):
    global id_counter
    for subdomain in os.listdir(root_directory):
        subdomain_path = os.path.join(root_directory, subdomain)

        if os.path.isdir(subdomain_path):
            print(f"Processing Subdomain: {subdomain}")
            for file in os.listdir(subdomain_path):
                file_path = os.path.join(subdomain_path, file)
                if file.endswith(".json"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            url = data['url']
                            content = data['content']
                            word_info = content_parser(content)
                            if word_info is None or len(word_info) == 0:
                                continue
                            id_counter += 1
                            urls.append(url)
                            update_inverted_index(word_info, id_counter)
                            if id_counter%BATCH_SIZE == 0:
                                save_inverted_index_by_partitions(inverted_index)
                                inverted_index.clear()
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        traceback.print_exc()

    if len(inverted_index) >0:
        save_inverted_index_by_partitions(inverted_index)

def generate_index_report(index_directory="indexes_by_partition", output_file="index_report.txt"):
    report_data = {
        "unique_tokens": 0,
        "indexed_documents": id_counter,
        "total_size_kb": 0.0
    }
    
    try:
        # Calculate statistics
        total_size_bytes = 0
        unique_tokens = set()
        
        if not os.path.exists(index_directory):
            raise FileNotFoundError(f"Index directory '{index_directory}' not found")
            
        # Process index files
        for filename in os.listdir(index_directory):
            if filename.startswith("index_") and filename.endswith(".json"):
                file_path = os.path.join(index_directory, filename)
                total_size_bytes += os.path.getsize(file_path)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    unique_tokens.update(data.keys())
        
        # Finalize calculations
        report_data["total_size_kb"] = round(total_size_bytes / 1024, 2)
        report_data["unique_tokens"] = len(unique_tokens)
        
        # Write to TXT file
        with open(output_file, 'w', encoding='utf-8') as report_file:
            report_file.write("=== Search Engine Index Report ===\n")
            report_file.write("-" * 40 + "\n")
            report_file.write(f"{'Unique Tokens':<25}{report_data['unique_tokens']:>15,}\n")
            report_file.write(f"{'Indexed Documents':<25}{report_data['indexed_documents']:>15,}\n")
            report_file.write(f"{'Total Index Size (KB)':<25}{report_data['total_size_kb']:>15,.2f}\n")
        
        print(f"Report generated successfully: {output_file}")
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return None

if __name__ == "__main__":
    create_files() # create INDEX DIR and 36 empty files
    root_directory = DATASET
    read_json_files(DATASET)
    store_urls(urls, URLS) # store the urls
    total_indexed_documents = len(urls)
    print(f"Total indexed documents: {total_indexed_documents}")
    idf = calculate_idf(total_indexed_documents, INDEX_DIR)
    with open(IDF, 'w', encoding='utf-8') as idf_file:
        json.dump(idf, idf_file)
    update_tf_idf_in_index(idf, INDEX_DIR)
   


