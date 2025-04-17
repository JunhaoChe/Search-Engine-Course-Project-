from config import OPT_INDEX_DIR, DATASET, DOCUMENTS_CONTENTS, OPT_INDEX, OPT_INDEX_OFFSET
from store_load import *

if __name__ == '__main__':
    store_doc(DATASET, DOCUMENTS_CONTENTS)
    optimize_inverted_index()
    optimize_inverted_index_plus_tf_idf(OPT_INDEX_DIR, INDEX_DIR)
    store_and_record_offset_for_optimized_index(OPT_INDEX_DIR, OPT_INDEX,OPT_INDEX_OFFSET)
