import os
import json
import traceback
from collections import defaultdict
from bs4 import BeautifulSoup


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
    'title': 2,
    'h1': 1.7,
    'h2': 1.5,
    'h3': 1.3,
    'strong': 1.2,
    'b': 1.2,
    'article' : 1,
    'p' : 1,
    'div' : 1,
    'li' : 1
}
base_path="indexes_by_partition"
inverted_index = {}
word_occurrence = {}
id_counter = 0
ids = {}
BATCH_SIZE = 1000
simhash_fingerprints = []

def generate_simhash(tokens):
    return Simhash(tokens).value
def find_near_duplicates(new_fingerprint, threshold=4):
    for fingerprint in simhash_fingerprints:
        hamming_distance = bin(new_fingerprint ^ fingerprint).count('1')
        if hamming_distance <= threshold:
            return True
    simhash_fingerprints.append(new_fingerprint)
    return False

def computeWordFrequencies(tokens):
    word_info = {}
    for token in tokens:
        if token not in word_info:
            word_info[token] = {"f": 1, "w": 1}
        else:
            word_info[token]["f"] += 1
    return word_info

def tokenizer(text):
    porter_stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    tokens = []
    token = ""

    for char in text:
        char = char.lower()
        if 'a' <= char <= 'z' or '0' <= char <= '9':
            token += char
        else:
            if token and token not in stop_words:
                token = porter_stemmer.stem(token)  # Apply stemming
                tokens.append(token)
            token = ''
    # Add last token if valid
    if token and token not in stop_words:
        token = porter_stemmer.stem(token)
        tokens.append(token)

    return tokens

def content_parser(content):
    if not content:
        return None
    soup = BeautifulSoup(content, 'html.parser')
    for tag in soup(['script', 'style', 'noscript', 'nav', 'form']):
        tag.decompose()
    all_tokens = []
    word_info = {}
    for tag, weight in TAG_WEIGHTS.items():
        for element in soup.find_all(tag):
            if not any(element in parent.contents for parent in element.find_parents(TAG_WEIGHTS.keys())):
                text = element.get_text(separator=" ").strip()
                if not text:
                    continue
                max_weight = weight
                for parent in element.find_parents(TAG_WEIGHTS.keys()):
                    parent_weight = TAG_WEIGHTS.get(parent.name, 1)
                    max_weight = max(max_weight, parent_weight)
                tokens = tokenizer(text)
                all_tokens.extend(tokens)
                for token in tokens:
                    if token in word_info:
                        word_info[token]["f"] += 1
                    else:
                        word_info[token] = {"f": 1, "w": max_weight}
    simhash_fingerprint = generate_simhash(all_tokens)
    if find_near_duplicates(simhash_fingerprint):
        return {}
    return word_info

def update_inverted_index(word_info, id):
    global inverted_index
    for token, attributes in word_info.items():
        if token in inverted_index:
            inverted_index[token].append({str(id): {"f": attributes["f"], "w": attributes["w"]}})
        else:
            inverted_index[token] = [{str(id): {"f": attributes["f"], "w": attributes["w"]}}]

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
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    try:
        for i in range(ord('a'), ord('z')+1):
            filepath = f"{base_path}/index_{chr(i)}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                pass
        for i in range(ord('0'), ord('9')+1):
            filepath = f"{base_path}/index_{chr(i)}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                pass
    except IOError as e:
        print(f"File creation has failed: {e}")


def save_inverted_index_by_partitions(inverted_index, base_path="indexes_by_partition"):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

  # Organize tokens into partitions based on their first character.
    partitioned = {}
    for token, postings in inverted_index.items():
        partition = get_partition_filename(token)
        if partition not in partitioned:
            partitioned[partition] = {}
        partitioned[partition][token] = postings
    # load existing data, update, and then write it back.
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

        # Merge new data with existing
        for token, postings in new_data.items():
            if token in existing_data:
                # Append new postings to the list.
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
                            ids[id_counter] = url
                            update_inverted_index(word_info, id_counter)
                            if id_counter%BATCH_SIZE == 0:
                                save_inverted_index_by_partitions(inverted_index)
                                inverted_index.clear()
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

                        traceback.print_exc()
    if len(inverted_index) >0:
        save_inverted_index_by_partitions(inverted_index)


if __name__ == "__main__":
    create_files()
    root_directory = "DEV"
    read_json_files(root_directory)
   


