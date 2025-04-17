import os

# Get the absolute path of the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # Move to the project root

# Define dataset paths
DATASET = os.path.join(BASE_DIR, "DEV")
INDEX_DIR = os.path.join(BASE_DIR, "indexes_by_partition")
OFFSET_DIR = os.path.join(BASE_DIR, "offsets")
DATA_DIR = os.path.join(BASE_DIR, "data")
URLS = os.path.join(BASE_DIR, "urls.txt")
OPT_INDEX_DIR = os.path.join(BASE_DIR, "optimized_inverted_indexes")
DOCUMENTS_CONTENTS = os.path.join(BASE_DIR, "contents.json")
OPT_INDEX = os.path.join(BASE_DIR, "opt_index.txt")
OPT_INDEX_OFFSET = os.path.join(BASE_DIR, "opt_offset.json")
IDF = os.path.join(BASE_DIR, "idf.json")

