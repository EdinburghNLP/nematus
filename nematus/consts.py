# DICT_SIZE = 29344
EMBEDDING_SIZE = 256
# DICT_SIZE = 30546
# DICT_SIZE = 29344
DICT_SIZE = 30648

# debias files
DEFINITIONAL_FILE = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/debiaswe/data/definitional_pairs.json"
GENDER_SPECIFIC_FILE = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/debiaswe/data/gender_specific_full.json"
PROFESSIONS_FILE = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/debiaswe/data/professions.json"
EQUALIZE_FILE = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/debiaswe/data/equalize_pairs.json"

# the file to which the debiased embedding table is saved at the end
DEBIASED_TARGET_FILE = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/debiaswe/embeddings/Nematus-hard-debiased.bin"

# the file to which the initial embedding table is pickled to after parsing the file written whn running translate
EMBEDDING_TABLE_FILE = "/cs/labs/gabis/bareluz/nematus_clean/nematus/embedding_table.bin"

# the file to which the initial embedding is written in the format of [word] [embedding]\n which is the format debiaswe uses. this is ready to be debiased
EMBEDDING_DEBIASWE_FILE = "/cs/labs/gabis/bareluz/nematus_clean/nematus/embedding_debiaswe.txt"

# the source english dictionary
# ENG_DICT_FILE = "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/train.clean.unesc.tok.tc.bpe.en.json"
# ENG_DICT_FILE = "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/20.07.21//train.clean.unesc.tok.tc.bpe.en.json"
ENG_DICT_FILE = "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_ru/30.11.20//train.clean.unesc.tok.tc.bpe.en.json"

# the path of the file that translate wrote the embedding table to. this file will be parsed and debiased
OUTPUT_TRANSLATE_FILE= "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/output_translate_ru.txt"

USE_DEBIASED = 0