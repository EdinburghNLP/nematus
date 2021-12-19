import json
import ast
#
# if __name__ == '__main__':
#     CONSTS_CONFIG = {"USE_DEBIASED":0, "LANGUAGE":0, "COLLECT_EMBEDDING_TABLE":0}
#     j =json.dumps(CONSTS_CONFIG)
#     with open("/cs/labs/gabis/bareluz/nematus_clean/nematus/consts_config.json","w") as f:
#         f.write(j)
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
        "EN_ANTI_MERGED": "/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/data/aggregates/en_ru_anti.en,txt",
        "ANTI_TRANSLATED_DEBIASED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/output/debiased_anti.out.tmp",
        "ANTI_TRANSLATED_NON_DEBIASED" : "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/output/non_debiased_anti.out.tmp",
        "ANTI_TRANSLATED_DEBIASED_PICKLE": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/output/debiased_anti.pickle",
        "ANTI_TRANSLATED_NON_DEBIASED_PICKLE": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/output/non_debiased_anti.pickle",
        "DEBIASED_EVAL": "/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/translations/nematus/en-ru-debiased.txt",
        "NON_DEBIASED_EVAL": "/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/translations/nematus//en-ru-non-debiased.txt",
        "BLEU_SOURCE_DATA": "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_ru/30.11.20/newstest2019-enru.unesc.tok.tc.bpe.en",
        "BLEU_GOLD_DATA": "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_ru/30.11.20/newstest2019-enru.unesc.tok.tc.bpe.ru",
        "TRANSLATED_DEBIASED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/output/debiased.out.tmp",
        "TRANSLATED_NON_DEBIASED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/output/non_debiased.out.tmp",
        "TRANSLATED_DEBIASED_PICKLE": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/evaluate/debiased.pickle",
        "TRANSLATED_NON_DEBIASED_PICKLE": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/evaluate/non_debiased.pickle",
        "BLEU_GOLD_DATA_FILTERED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/evaluate/gold.txt",

        },
    Language.GERMAN:
        {
        "DICT_SIZE": 29344,
        "ENG_DICT_FILE": "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/train.clean.unesc.tok.tc.bpe.en.json",
        "OUTPUT_TRANSLATE_FILE": "/cs/labs/gabis/bareluz/nematus_clean/nematus/en-de/debias/output_translate_de.txt",
        "EMBEDDING_TABLE_FILE": "/cs/labs/gabis/bareluz/nematus_clean/nematus/en-de/debias/embedding_table_de.bin",
        "EMBEDDING_DEBIASWE_FILE": "/cs/labs/gabis/bareluz/nematus_clean/nematus/en-de/debias/embedding_debiaswe_de.txt",
        "DEBIASED_TARGET_FILE": "/cs/labs/gabis/bareluz/nematus_clean/nematus/en-de/debias/Nematus-hard-debiased-de.bin",
        "EN_ANTI_PARSED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/anti.en",
        "EN_ANTI_MERGED": "/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/data/aggregates/en_de_anti.en.txt",
        "ANTI_TRANSLATED_DEBIASED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/output/debiased_anti.out.tmp",
        "ANTI_TRANSLATED_NON_DEBIASED" : "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/output/non_debiased_anti.out.tmp",
        "ANTI_TRANSLATED_DEBIASED_PICKLE": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/output/debiased_anti.pickle",
        "ANTI_TRANSLATED_NON_DEBIASED_PICKLE": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/output/non_debiased_anti.pickle",
        "DEBIASED_EVAL": "/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/translations/nematus/en-de-debiased.txt",
        "NON_DEBIASED_EVAL": "/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/translations/nematus//en-de-non-debiased.txt",
        "BLEU_SOURCE_DATA": "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/newstest2012.unesc.tok.tc.bpe.en",
        "BLEU_GOLD_DATA": "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/newstest2012.unesc.tok.tc.bpe.de",
        "TRANSLATED_DEBIASED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/output/debiased.out.tmp",
        "TRANSLATED_NON_DEBIASED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/output/non_debiased.out.tmp",
        "TRANSLATED_DEBIASED_PICKLE": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/evaluate/debiased.pickle",
        "TRANSLATED_NON_DEBIASED_PICKLE": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/evaluate/non_debiased.pickle",
        "BLEU_GOLD_DATA_FILTERED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/evaluate/gold.txt",
        },
    Language.HEBREW:
        {
            "DICT_SIZE": 30545,
            "ENG_DICT_FILE": "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/20.07.21//train.clean.unesc.tok.tc.bpe.en.json",
            "OUTPUT_TRANSLATE_FILE": "/cs/labs/gabis/bareluz/nematus_clean/nematus/en-he/debias/output_translate_he.txt",
            "EMBEDDING_TABLE_FILE": "/cs/labs/gabis/bareluz/nematus_clean/nematus/en-he/debias/embedding_table_he.bin",
            "EMBEDDING_DEBIASWE_FILE": "/cs/labs/gabis/bareluz/nematus_clean/nematus/en-he/debias/embedding_debiaswe_he.txt",
            "DEBIASED_TARGET_FILE": "/cs/labs/gabis/bareluz/nematus_clean/nematus/en-he/debias/Nematus-hard-debiased-he.bin",
            "EN_ANTI_PARSED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/anti.en",
            "EN_ANTI_MERGED": "/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/data/aggregates/en_he_anti.en.txt",
            "ANTI_TRANSLATED_DEBIASED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/output/debiased_anti.out.tmp",
            "ANTI_TRANSLATED_NON_DEBIASED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/output/non_debiased_anti.out.tmp",
            "ANTI_TRANSLATED_DEBIASED_PICKLE": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/output/debiased_anti.pickle",
            "ANTI_TRANSLATED_NON_DEBIASED_PICKLE": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/output/non_debiased_anti.pickle",
            "DEBIASED_EVAL": "/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/translations/nematus/en-he-debiased.txt",
            "NON_DEBIASED_EVAL": "/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/translations/nematus//en-he-non-debiased.txt",
            "BLEU_SOURCE_DATA": "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/20.07.21//dev.unesc.tok.tc.bpe.en",
            "BLEU_GOLD_DATA": "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/20.07.21//dev.unesc.tok.bpe.he",
            "TRANSLATED_DEBIASED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/output/debiased.out.tmp",
            "TRANSLATED_NON_DEBIASED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/output/non_debiased.out.tmp",
            "TRANSLATED_DEBIASED_PICKLE": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/evaluate/debiased.pickle",
            "TRANSLATED_NON_DEBIASED_PICKLE": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/evaluate/non_debiased.pickle",
            "BLEU_GOLD_DATA_FILTERED": "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/evaluate/gold.txt",
        }
}

def parse_config(config_str):
    return ast.literal_eval(config_str)

def get_u_l_c_p(config_str):
    config = parse_config(config_str)
    USE_DEBIASED = config["USE_DEBIASED"]
    LANGUAGE = config["LANGUAGE"]
    COLLECT_EMBEDDING_TABLE = config["COLLECT_EMBEDDING_TABLE"]
    PRINT_LINE_NUMS = config["PRINT_LINE_NUMS"]
    return USE_DEBIASED, LANGUAGE, COLLECT_EMBEDDING_TABLE, PRINT_LINE_NUMS


def get_debias_files_from_config(config_str):
    config = parse_config(config_str)
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



def get_evaluate_gender_files(config_str):
    config = parse_config(config_str)
    # the path to Gabi's anti sentences after they were separated from the rest of the data
    EN_ANTI_PARSED = param_dict[Language(int(config['LANGUAGE']))]["EN_ANTI_PARSED"]
    EN_ANTI_MERGED = param_dict[Language(int(config['LANGUAGE']))]["EN_ANTI_MERGED"]
    ANTI_TRANSLATED_DEBIASED = param_dict[Language(int(config['LANGUAGE']))]["ANTI_TRANSLATED_DEBIASED"]
    ANTI_TRANSLATED_DEBIASED_PICKLE = param_dict[Language(int(config['LANGUAGE']))]["ANTI_TRANSLATED_DEBIASED_PICKLE"]
    # the translation of Gabi's anti sentences using non debiased embedding table
    ANTI_TRANSLATED_NON_DEBIASED = param_dict[Language(int(config['LANGUAGE']))]["ANTI_TRANSLATED_NON_DEBIASED"]
    ANTI_TRANSLATED_NON_DEBIASED_PICKLE = param_dict[Language(int(config['LANGUAGE']))]["ANTI_TRANSLATED_NON_DEBIASED_PICKLE"]
    # file prepared to evaluation in the form of source_sentence ||| translated_sentence. translated using debiased embedding table
    DEBIASED_EVAL = param_dict[Language(int(config['LANGUAGE']))]["DEBIASED_EVAL"]
    # file prepared to evaluation in the form of source_sentence ||| translated_sentence. translated using non debiased embedding table
    NON_DEBIASED_EVAL = param_dict[Language(int(config['LANGUAGE']))]["NON_DEBIASED_EVAL"]
    # the path to Gabi's anti sentences before they were separated from the rest of the data
    EN_ANTI = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en_anti.txt"
    EN_ANTI_MT_GENDER = "/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/data/aggregates/en_anti.txt"
    return EN_ANTI_PARSED, ANTI_TRANSLATED_DEBIASED, ANTI_TRANSLATED_NON_DEBIASED, DEBIASED_EVAL, NON_DEBIASED_EVAL,\
           EN_ANTI, EN_ANTI_MERGED, ANTI_TRANSLATED_DEBIASED_PICKLE, ANTI_TRANSLATED_NON_DEBIASED_PICKLE, EN_ANTI_MT_GENDER

def get_evaluate_translate_files(config_str):
    config = parse_config(config_str)
    # data of sentences for Bleu evaluation
    BLEU_SOURCE_DATA = param_dict[Language(int(config['LANGUAGE']))]["BLEU_SOURCE_DATA"]
    # data of the gold translation sentences for Bleu evaluation
    BLEU_GOLD_DATA = param_dict[Language(int(config['LANGUAGE']))]["BLEU_GOLD_DATA"]
    TRANSLATED_DEBIASED = param_dict[Language(int(config['LANGUAGE']))]["TRANSLATED_DEBIASED"]
    TRANSLATED_NON_DEBIASED = param_dict[Language(int(config['LANGUAGE']))]["TRANSLATED_NON_DEBIASED"]
    # TRANSLATED_NON_DEBIASED2 = param_dict[Language(int(config['LANGUAGE']))]["TRANSLATED_NON_DEBIASED2"]
    TRANSLATED_DEBIASED_PICKLE = param_dict[Language(int(config['LANGUAGE']))]["TRANSLATED_DEBIASED_PICKLE"]
    TRANSLATED_NON_DEBIASED_PICKLE = param_dict[Language(int(config['LANGUAGE']))]["TRANSLATED_NON_DEBIASED_PICKLE"]
    BLEU_GOLD_DATA_FILTERED = param_dict[Language(int(config['LANGUAGE']))]["BLEU_GOLD_DATA_FILTERED"]
    return BLEU_SOURCE_DATA, BLEU_GOLD_DATA, TRANSLATED_DEBIASED, TRANSLATED_NON_DEBIASED,\
           TRANSLATED_DEBIASED_PICKLE, TRANSLATED_NON_DEBIASED_PICKLE, BLEU_GOLD_DATA_FILTERED

#################debiaswe files#################
DEFINITIONAL_FILE = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/debiaswe/data/definitional_pairs.json"
GENDER_SPECIFIC_FILE = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/debiaswe/data/gender_specific_full.json"
PROFESSIONS_FILE = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/debiaswe/data/professions.json"
EQUALIZE_FILE = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/debiaswe/data/equalize_pairs.json"

