import json
#
# if __name__ == '__main__':
#     CONSTS_CONFIG = {"USE_DEBIASED":0, "LANGUAGE":0, "COLLECT_EMBEDDING_TABLE":0}
#     j =json.dumps(CONSTS_CONFIG)
#     with open("/cs/labs/gabis/bareluz/nematus_clean/nematus/consts_config.json","w") as f:
#         f.write(j)
print("running consts")
from enum import Enum
class Language(Enum):
    RUSSIAN = 0
    GERMAN = 1
    HEBREW = 2

EMBEDDING_SIZE = 256
param_dict = {
    Language.RUSSIAN:
        {
        "DICT_SIZE": 30648,
        "ENG_DICT_FILE": "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_ru/30.11.20//train.clean.unesc.tok.tc.bpe.en.json",
        "OUTPUT_TRANSLATE_FILE": "/cs/labs/gabis/bareluz/nematus_clean/nematus/en-ru/debias/output_translate_ru.txt",
        "EMBEDDING_TABLE_FILE": "/cs/labs/gabis/bareluz/nematus_clean/nematus/en-ru/debias/embedding_table_ru.bin",
        "EMBEDDING_DEBIASWE_FILE": "/cs/labs/gabis/bareluz/nematus_clean/nematus/en-ru/debias/embedding_debiaswe_ru.txt",
        "DEBIASED_TARGET_FILE": "/cs/labs/gabis/bareluz/nematus_clean/nematus/en-ru/debias/Nematus-hard-debiased-ru.bin",
        "EN_ANTI_PARSED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/anti.en",
        "ANTI_TRANSLATED_DEBIASED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/output/debiased_anti.out.tmp",
        "ANTI_TRANSLATED_NON_DEBIASED" : "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/output/non_debiased_anti.out.tmp",
        "DEBIASED_EVAL": "/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/translations/nematus/en-ru-debiased.txt",
        "NON_DEBIASED_EVAL": "/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/translations/nematus//en-ru-non-debiased.txt",
        "TRANSLATE_SEPARATE_BAD_LINES": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/evaluate/output_translate_ru_separate_bad_lines.txt",
        "BLEU_SOURCE_DATA": "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_ru/30.11.20/newstest2019-enru.unesc.tok.tc.bpe.en",
        "BLEU_GOLD_DATA": "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_ru/30.11.20/newstest2019-enru.unesc.tok.tc.bpe.ru",
        "BLEU_SOURCE_DATA_FILTERED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/evaluate/newstest2019-enru.unesc.tok.tc.bpe.en",
        "BLEU_GOLD_DATA_FILTERED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/evaluate/newstest2019-enru.unesc.tok.tc.bpe.ru",
        "BLEU_SOURCE_DATA_FILTERED2": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/evaluate/newstest2019-enru2.unesc.tok.tc.bpe.en",
        "BLEU_GOLD_DATA_FILTERED2": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/evaluate/newstest2019-enru2.unesc.tok.tc.bpe.ru",
        "TRANSLATED_DEBIASED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/output/debiased.out.tmp",
        "TRANSLATED_NON_DEBIASED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/output/non_debiased.out.tmp",
        "TRANSLATED_NON_DEBIASED2": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/output/non_debiased1.out.tmp",
        "TRANSLATED_DEBIASED_PICKLE": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/output/debiased.pickle",
        "TRANSLATED_NON_DEBIASED_PICKLE": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/output/non_debiased.pickle",
        "BLEU_GOLD_DATA_FILTERED_PICKLE": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/evaluate/newstest2019-enru2.pickle",

        },
    Language.GERMAN:
        {
        "DICT_SIZE": 29344,
        "ENG_DICT_FILE": "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/train.clean.unesc.tok.tc.bpe.en.json",
        "OUTPUT_TRANSLATE_FILE": "/cs/labs/gabis/bareluz/nematus_clean/nematus/en-de-rev/debias/output_translate_de.txt",
        "EMBEDDING_TABLE_FILE": "/cs/labs/gabis/bareluz/nematus_clean/nematus/en-de-rev/debias/embedding_table_de.bin",
        "EMBEDDING_DEBIASWE_FILE": "/cs/labs/gabis/bareluz/nematus_clean/nematus/en-de-rev/debias/embedding_debiaswe_de.txt",
        "DEBIASED_TARGET_FILE": "/cs/labs/gabis/bareluz/nematus_clean/nematus/en-de-rev/debias/Nematus-hard-debiased-de.bin",
        "EN_ANTI_PARSED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/anti.en",
        "ANTI_TRANSLATED_DEBIASED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/output/debiased_anti.out.tmp",
        "ANTI_TRANSLATED_NON_DEBIASED" : "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/output/non_debiased_anti.out.tmp",
        "DEBIASED_EVAL": "/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/translations/nematus/en-de-debiased.txt",
        "NON_DEBIASED_EVAL": "/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/translations/nematus//en-de-non-debiased.txt",
        "TRANSLATE_SEPARATE_BAD_LINES": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/evaluate/output_translate_de_separate_bad_lines.txt",
        "BLEU_SOURCE_DATA": "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/train.clean.unesc.tok.tc.bpe.en",
        "BLEU_GOLD_DATA": "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/train.clean.unesc.tok.tc.bpe.de",
        "BLEU_SOURCE_DATA_FILTERED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/evaluate/train.clean.unesc.tok.tc.bpe.en",
        "BLEU_GOLD_DATA_FILTERED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/evaluate/train.clean.unesc.tok.tc.bpe.de",
        "BLEU_SOURCE_DATA_FILTERED2": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/evaluate/train2.clean.unesc.tok.tc.bpe.en",
        "BLEU_GOLD_DATA_FILTERED2": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/evaluate/train2.clean.unesc.tok.tc.bpe.de",
        "TRANSLATED_DEBIASED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/output/debiased.out.tmp",
        "TRANSLATED_NON_DEBIASED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/output/non_debiased.out.tmp",
        "TRANSLATED_NON_DEBIASED2": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/output/non_debiased1.out.tmp",
        "TRANSLATED_DEBIASED_PICKLE": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/output/debiased.pickle",
        "TRANSLATED_NON_DEBIASED_PICKLE": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/output/non_debiased.pickle",
        "BLEU_GOLD_DATA_FILTERED_PICKLE": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/evaluate/train2.pickle",
        },
    Language.HEBREW:
        {
            "DICT_SIZE": 30545,
            "ENG_DICT_FILE": "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/20.07.21//train.clean.unesc.tok.tc.bpe.en.json",
            "OUTPUT_TRANSLATE_FILE": "/cs/labs/gabis/bareluz/nematus_clean/nematus/en-he/debias/output_translate_he.txt",
            "EMBEDDING_TABLE_FILE": "/cs/labs/gabis/bareluz/nematus_clean/nematus/en-he/debias/embedding_table_he.bin",
            "EMBEDDING_DEBIASWE_FILE": "/cs/labs/gabis/bareluz/nematus_clean/nematus/en-he/debias/embedding_debiaswe_he.txt",
            "DEBIASED_TARGET_FILE": "/cs/labs/gabis/bareluz/nematus_clean/nematus/en-he/debias/Nematus-hard-debiased-he.bin",
            "EN_ANTI_PARSED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/anti.en",
            "ANTI_TRANSLATED_DEBIASED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/output/debiased_anti.out.tmp",
            "ANTI_TRANSLATED_NON_DEBIASED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/output/non_debiased_anti.out.tmp",
            "DEBIASED_EVAL": "/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/translations/nematus/en-he-debiased.txt",
            "NON_DEBIASED_EVAL": "/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/translations/nematus//en-he-non-debiased.txt",
            "TRANSLATE_SEPARATE_BAD_LINES": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/evaluate/output_translate_he_separate_bad_lines.txt",
            "BLEU_SOURCE_DATA": "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/20.07.21//train.clean.unesc.tok.tc.bpe.en",
            "BLEU_GOLD_DATA": "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/20.07.21//train.clean.unesc.tok.tc.bpe.he",
            "BLEU_SOURCE_DATA_FILTERED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/evaluate/train.clean.unesc.tok.tc.bpe.en",
            "BLEU_GOLD_DATA_FILTERED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/evaluate/train.clean.unesc.tok.tc.bpe.he",
            "BLEU_SOURCE_DATA_FILTERED2": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/evaluate/train2.clean.unesc.tok.tc.bpe.en",
            "BLEU_GOLD_DATA_FILTERED2": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/evaluate/train2.clean.unesc.tok.tc.bpe.he",
            "TRANSLATED_DEBIASED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/output/debiased.out.tmp",
            "TRANSLATED_NON_DEBIASED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/output/non_debiased.out.tmp",
            "TRANSLATED_NON_DEBIASED2": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/output/non_debiased1.out.tmp",
            "TRANSLATED_DEBIASED_PICKLE": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/output/debiased.pickle",
            "TRANSLATED_NON_DEBIASED_PICKLE": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/output/non_debiased.pickle",
            "BLEU_GOLD_DATA_FILTERED_PICKLE": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/evaluate/train2.pickle",
        }
}

def load_config(config_file):
    f = open(config_file,'r')
    config = json.load(f)
    f.close()
    return config
def get_u_l_c(config_file):
    config = load_config(config_file)
    USE_DEBIASED = config["USE_DEBIASED"]
    LANGUAGE = config["LANGUAGE"]
    COLLECT_EMBEDDING_TABLE = config["COLLECT_EMBEDDING_TABLE"]
    return USE_DEBIASED, LANGUAGE, COLLECT_EMBEDDING_TABLE

# #################debias files#################
# DICT_SIZE  = param_dict[LANGUAGE]["DICT_SIZE"]
# # the source english dictionary
# ENG_DICT_FILE = param_dict[LANGUAGE]["ENG_DICT_FILE"]
# # the path of the file that translate wrote the embedding table to. this file will be parsed and debiased
# OUTPUT_TRANSLATE_FILE = param_dict[LANGUAGE]["OUTPUT_TRANSLATE_FILE"]
# # the file to which the initial embedding table is pickled to after parsing the file written whn running translate
# EMBEDDING_TABLE_FILE = param_dict[LANGUAGE]["EMBEDDING_TABLE_FILE"]
# # the file to which the initial (non debiased) embedding is written in the format of [word] [embedding]\n which is the format debiaswe uses. this is ready to be debiased
# EMBEDDING_DEBIASWE_FILE = param_dict[LANGUAGE]["EMBEDDING_DEBIASWE_FILE"]
# # the file to which the debiased embedding table is saved at the end
# DEBIASED_TARGET_FILE = param_dict[LANGUAGE]["DEBIASED_TARGET_FILE"]


def get_debias_files_from_config(config_file):
    config = load_config(config_file)
    DICT_SIZE = param_dict[Language(int(config['LANGUAGE']))]["DICT_SIZE"]
    # the source english dictionary
    ENG_DICT_FILE = param_dict[Language(int(config['LANGUAGE']))]["ENG_DICT_FILE"]
    # the path of the file that translate wrote the embedding table to. this file will be parsed and debiased
    OUTPUT_TRANSLATE_FILE = param_dict[Language(int(config['LANGUAGE']))]["OUTPUT_TRANSLATE_FILE"]
    # the file to which the initial embedding table is pickled to after parsing the file written whn running translate
    EMBEDDING_TABLE_FILE = param_dict[Language(int(config['LANGUAGE']))]["EMBEDDING_TABLE_FILE"]
    # the file to which the initial (non debiased) embedding is written in the format of [word] [embedding]\n which is the format debiaswe uses. this is ready to be debiased
    EMBEDDING_DEBIASWE_FILE = param_dict[Language(int(config['LANGUAGE']))]["EMBEDDING_DEBIASWE_FILE"]
    # the file to which the debiased embedding table is saved at the end
    DEBIASED_TARGET_FILE = param_dict[Language(int(config['LANGUAGE']))]["DEBIASED_TARGET_FILE"]
    return DICT_SIZE, ENG_DICT_FILE, OUTPUT_TRANSLATE_FILE, EMBEDDING_TABLE_FILE, EMBEDDING_DEBIASWE_FILE, DEBIASED_TARGET_FILE

# #################evaluate gender files#################
# # the path to Gabi's anti sentences after they were separated from the rest of the data
# EN_ANTI_PARSED = param_dict[LANGUAGE]["EN_ANTI_PARSED"]
# ANTI_TRANSLATED_DEBIASED = param_dict[LANGUAGE]["ANTI_TRANSLATED_DEBIASED"]
# # the translation of Gabi's anti sentences using non debiased embedding table
# ANTI_TRANSLATED_NON_DEBIASED = param_dict[LANGUAGE]["ANTI_TRANSLATED_NON_DEBIASED"]
# # file prepared to evaluation in the form of source_sentence ||| translated_sentence. translated using debiased embedding table
# DEBIASED_EVAL = param_dict[LANGUAGE]["DEBIASED_EVAL"]
# # file prepared to evaluation in the form of source_sentence ||| translated_sentence. translated using non debiased embedding table
# NON_DEBIASED_EVAL = param_dict[LANGUAGE]["NON_DEBIASED_EVAL"]
# # the path to Gabi's anti sentences before they were separated from the rest of the data
# EN_ANTI = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en_anti.txt"

def get_evaluate_gender_files(config_file):
    config = load_config(config_file)
    # the path to Gabi's anti sentences after they were separated from the rest of the data
    EN_ANTI_PARSED = param_dict[Language(int(config['LANGUAGE']))]["EN_ANTI_PARSED"]
    ANTI_TRANSLATED_DEBIASED = param_dict[Language(int(config['LANGUAGE']))]["ANTI_TRANSLATED_DEBIASED"]
    # the translation of Gabi's anti sentences using non debiased embedding table
    ANTI_TRANSLATED_NON_DEBIASED = param_dict[Language(int(config['LANGUAGE']))]["ANTI_TRANSLATED_NON_DEBIASED"]
    # file prepared to evaluation in the form of source_sentence ||| translated_sentence. translated using debiased embedding table
    DEBIASED_EVAL = param_dict[Language(int(config['LANGUAGE']))]["DEBIASED_EVAL"]
    # file prepared to evaluation in the form of source_sentence ||| translated_sentence. translated using non debiased embedding table
    NON_DEBIASED_EVAL = param_dict[Language(int(config['LANGUAGE']))]["NON_DEBIASED_EVAL"]
    # the path to Gabi's anti sentences before they were separated from the rest of the data
    EN_ANTI = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en_anti.txt"
    return EN_ANTI_PARSED, ANTI_TRANSLATED_DEBIASED, ANTI_TRANSLATED_NON_DEBIASED, DEBIASED_EVAL, NON_DEBIASED_EVAL, EN_ANTI

# #################evaluate translate files#################
# # file to which the lines that made the "translate" run crash are written
# TRANSLATE_SEPARATE_BAD_LINES = param_dict[LANGUAGE]["TRANSLATE_SEPARATE_BAD_LINES"]
# # data of sentences for Bleu evaluation
# BLEU_SOURCE_DATA = param_dict[LANGUAGE]["BLEU_SOURCE_DATA"]
# # data of the gold translation sentences for Bleu evaluation
# BLEU_GOLD_DATA = param_dict[LANGUAGE]["BLEU_GOLD_DATA"]
# BLEU_SOURCE_DATA_FILTERED = param_dict[LANGUAGE]["BLEU_SOURCE_DATA_FILTERED"]
# BLEU_GOLD_DATA_FILTERED =param_dict[LANGUAGE]["BLEU_GOLD_DATA_FILTERED"]
# BLEU_SOURCE_DATA_FILTERED2 = param_dict[LANGUAGE]["BLEU_SOURCE_DATA_FILTERED2"]
# BLEU_GOLD_DATA_FILTERED2 =param_dict[LANGUAGE]["BLEU_GOLD_DATA_FILTERED2"]
# TRANSLATED_DEBIASED = param_dict[LANGUAGE]["TRANSLATED_DEBIASED"]
# TRANSLATED_NON_DEBIASED =param_dict[LANGUAGE]["TRANSLATED_NON_DEBIASED"]
# TRANSLATED_NON_DEBIASED2 =param_dict[LANGUAGE]["TRANSLATED_NON_DEBIASED2"]
# TRANSLATED_DEBIASED_PICKLE =param_dict[LANGUAGE]["TRANSLATED_DEBIASED_PICKLE"]
# TRANSLATED_NON_DEBIASED_PICKLE =param_dict[LANGUAGE]["TRANSLATED_NON_DEBIASED_PICKLE"]
# BLEU_GOLD_DATA_FILTERED_PICKLE =param_dict[LANGUAGE]["BLEU_GOLD_DATA_FILTERED_PICKLE"]

def get_evaluate_translate_files(config_file):
    config = load_config(config_file)
    # file to which the lines that made the "translate" run crash are written
    TRANSLATE_SEPARATE_BAD_LINES = param_dict[Language(int(config['LANGUAGE']))]["TRANSLATE_SEPARATE_BAD_LINES"]
    # data of sentences for Bleu evaluation
    BLEU_SOURCE_DATA = param_dict[Language(int(config['LANGUAGE']))]["BLEU_SOURCE_DATA"]
    # data of the gold translation sentences for Bleu evaluation
    BLEU_GOLD_DATA = param_dict[Language(int(config['LANGUAGE']))]["BLEU_GOLD_DATA"]
    BLEU_SOURCE_DATA_FILTERED = param_dict[Language(int(config['LANGUAGE']))]["BLEU_SOURCE_DATA_FILTERED"]
    BLEU_GOLD_DATA_FILTERED = param_dict[Language(int(config['LANGUAGE']))]["BLEU_GOLD_DATA_FILTERED"]
    BLEU_SOURCE_DATA_FILTERED2 = param_dict[Language(int(config['LANGUAGE']))]["BLEU_SOURCE_DATA_FILTERED2"]
    BLEU_GOLD_DATA_FILTERED2 = param_dict[Language(int(config['LANGUAGE']))]["BLEU_GOLD_DATA_FILTERED2"]
    TRANSLATED_DEBIASED = param_dict[Language(int(config['LANGUAGE']))]["TRANSLATED_DEBIASED"]
    TRANSLATED_NON_DEBIASED = param_dict[Language(int(config['LANGUAGE']))]["TRANSLATED_NON_DEBIASED"]
    TRANSLATED_NON_DEBIASED2 = param_dict[Language(int(config['LANGUAGE']))]["TRANSLATED_NON_DEBIASED2"]
    TRANSLATED_DEBIASED_PICKLE = param_dict[Language(int(config['LANGUAGE']))]["TRANSLATED_DEBIASED_PICKLE"]
    TRANSLATED_NON_DEBIASED_PICKLE = param_dict[Language(int(config['LANGUAGE']))]["TRANSLATED_NON_DEBIASED_PICKLE"]
    BLEU_GOLD_DATA_FILTERED_PICKLE = param_dict[Language(int(config['LANGUAGE']))]["BLEU_GOLD_DATA_FILTERED_PICKLE"]
    return TRANSLATE_SEPARATE_BAD_LINES, BLEU_SOURCE_DATA, BLEU_GOLD_DATA,BLEU_SOURCE_DATA_FILTERED,  BLEU_GOLD_DATA_FILTERED, \
           BLEU_SOURCE_DATA_FILTERED2, BLEU_GOLD_DATA_FILTERED2, TRANSLATED_DEBIASED, TRANSLATED_NON_DEBIASED, TRANSLATED_NON_DEBIASED2,\
           TRANSLATED_DEBIASED_PICKLE, TRANSLATED_NON_DEBIASED_PICKLE, BLEU_GOLD_DATA_FILTERED_PICKLE

#################debiaswe files#################
DEFINITIONAL_FILE = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/debiaswe/data/definitional_pairs.json"
GENDER_SPECIFIC_FILE = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/debiaswe/data/gender_specific_full.json"
PROFESSIONS_FILE = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/debiaswe/data/professions.json"
EQUALIZE_FILE = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/debiaswe/data/equalize_pairs.json"

