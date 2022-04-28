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


LANGUAGE_STR_MAP = {Language.RUSSIAN: "ru", Language.GERMAN: "de", Language.HEBREW: "he"}


class DebiasMethod(Enum):
    BOLUKBASY = 0
    NULL_IT_OUT = 1


EMBEDDING_SIZE = 256
NEMATUS_HOME = "/cs/labs/gabis/bareluz/nematus_clean/nematus/"
PREPROCESS_HOME = "/cs/snapless/oabend/borgr/SSMT/preprocess/data/"
MT_GENDER_HOME = "/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/"
param_dict = {
    Language.RUSSIAN:
        {
            "DICT_SIZE": 30648,
            "ENG_DICT_FILE": PREPROCESS_HOME + "en_ru/30.11.20//train.clean.unesc.tok.tc.bpe.en.json",
            "BLEU_SOURCE_DATA": PREPROCESS_HOME + "en_ru/30.11.20/newstest2019-enru.unesc.tok.tc.bpe.en",
            "BLEU_GOLD_DATA": PREPROCESS_HOME + "en_ru/30.11.20/newstest2019-enru.unesc.tok.tc.bpe.ru",

        },
    Language.GERMAN:
        {
            "DICT_SIZE": 29344,
            "ENG_DICT_FILE": PREPROCESS_HOME + "en_de/5.8/train.clean.unesc.tok.tc.bpe.en.json",
            "BLEU_SOURCE_DATA": PREPROCESS_HOME + "en_de/5.8/newstest2012.unesc.tok.tc.bpe.en",
            "BLEU_GOLD_DATA": PREPROCESS_HOME + "en_de/5.8/newstest2012.unesc.tok.tc.bpe.de",
        },
    Language.HEBREW:
        {
            "DICT_SIZE": 30545,
            "ENG_DICT_FILE": PREPROCESS_HOME + "en_he/20.07.21//train.clean.unesc.tok.tc.bpe.en.json",
            "BLEU_SOURCE_DATA": PREPROCESS_HOME + "en_he/20.07.21//dev.unesc.tok.tc.bpe.en",
            "BLEU_GOLD_DATA": PREPROCESS_HOME + "en_he/20.07.21//dev.unesc.tok.bpe.he",
        }
}


def parse_config(config_str):
    return ast.literal_eval(config_str)


def get_basic_configurations(config_str):
    config = parse_config(config_str)
    USE_DEBIASED = config["USE_DEBIASED"]
    LANGUAGE = config["LANGUAGE"]
    COLLECT_EMBEDDING_TABLE = config["COLLECT_EMBEDDING_TABLE"]
    DEBIAS_METHOD = config["DEBIAS_METHOD"]

    return USE_DEBIASED, LANGUAGE, COLLECT_EMBEDDING_TABLE, DEBIAS_METHOD


def get_debias_files_from_config(config_str):
    config = parse_config(config_str)
    lang = LANGUAGE_STR_MAP[Language(config['LANGUAGE'])]
    debias_method = str(config['DEBIAS_METHOD'])

    DICT_SIZE = param_dict[Language(int(config['LANGUAGE']))]["DICT_SIZE"]

    # the source english dictionary
    ENG_DICT_FILE = param_dict[Language(int(config['LANGUAGE']))]["ENG_DICT_FILE"]

    # the path of the file that translate wrote the embedding table to. this file will be parsed and debiased
    OUTPUT_TRANSLATE_FILE = NEMATUS_HOME + "en-" + lang + "/debias/output_translate_" + lang + ".txt"

    # the file to which the initial embedding table is pickled to after parsing the file written when running translate
    EMBEDDING_TABLE_FILE = NEMATUS_HOME + "en-" + lang + "/debias/embedding_table_" + lang + ".bin"

    # the file to which the initial (non debiased) embedding is written in the format of [word] [embedding]\n which is the format debiaswe uses. this is ready to be debiased
    EMBEDDING_DEBIASWE_FILE = NEMATUS_HOME + "en-" + lang + "/debias/embedding_debiaswe_" + lang + ".txt"

    # the file to which the debiased embedding table is saved at the end
    DEBIASED_EMBEDDING = NEMATUS_HOME + "en-" + lang + "/debias/Nematus-hard-debiased-" + lang + "-"+debias_method+".txt"

    SANITY_CHECK__FILE = NEMATUS_HOME + "en-" + lang + "/debias/sanity_check.csv"

    return DICT_SIZE, ENG_DICT_FILE, OUTPUT_TRANSLATE_FILE, EMBEDDING_TABLE_FILE, EMBEDDING_DEBIASWE_FILE, DEBIASED_EMBEDDING, SANITY_CHECK__FILE


def get_evaluate_gender_files(config_str):
    config = parse_config(config_str)
    lang = LANGUAGE_STR_MAP[Language(config['LANGUAGE'])]
    debias_method = str(config['DEBIAS_METHOD'])
    # the translations of anti sentences, using the debiased embedding table, with source line nums printed
    ANTI_TRANSLATED_DEBIASED = NEMATUS_HOME + "en-" + lang + "/output/debiased_anti_"+debias_method+".out.tmp"

    # the translations of anti sentences, using the non debiased embedding table, with source line nums printed
    ANTI_TRANSLATED_NON_DEBIASED = NEMATUS_HOME + "en-" + lang + "/output/non_debiased_anti_"+debias_method+".out.tmp"

    # # the translations of anti sentences, using the debiased embedding table, after filtering lines that are cummon with the non debiased translation
    # ANTI_TRANSLATED_DEBIASED_MERGED = NEMATUS_HOME + "en-" + lang + "/output/debiased_anti_"+debias_method+".txt"

    # # the translations of anti sentences, using the non debiased embedding table, after filtering lines that are cummon with the debiased translation
    # ANTI_TRANSLATED_NON_DEBIASED_MERGED = NEMATUS_HOME + "en-" + lang + "/output/non_debiased_anti_"+debias_method+".txt"

    # the full anti sentences in english (in the format <gender> <profession location> <sentence> <profession>)
    EN_ANTI_MT_GENDER = MT_GENDER_HOME + "data/aggregates/en_anti.txt"

    # # the source sentences (in english) after filtering lines that were not translated (in the format <gender> <profession location> <sentence> <profession>)
    # EN_ANTI_MERGED = MT_GENDER_HOME + "data/aggregates/en_" + lang + "_anti_"+debias_method+".en.txt"

    # file prepared to evaluation in the form of source_sentence ||| translated_sentence. translated using debiased embedding table
    DEBIASED_EVAL = MT_GENDER_HOME + "translations/nematus/en-" + lang + "-debiased-"+debias_method+".txt"

    # file prepared to evaluation in the form of source_sentence ||| translated_sentence. translated using non debiased embedding table
    NON_DEBIASED_EVAL = MT_GENDER_HOME + "translations/nematus//en-" + lang + "-non-debiased-"+debias_method+".txt"



    return ANTI_TRANSLATED_DEBIASED, ANTI_TRANSLATED_NON_DEBIASED, DEBIASED_EVAL, NON_DEBIASED_EVAL, EN_ANTI_MT_GENDER


def get_evaluate_translation_files(config_str):
    config = parse_config(config_str)
    lang = LANGUAGE_STR_MAP[Language(config['LANGUAGE'])]
    debias_method = str(config['DEBIAS_METHOD'])

    # data of sentences for Bleu evaluation
    BLEU_SOURCE_DATA = param_dict[Language(int(config['LANGUAGE']))]["BLEU_SOURCE_DATA"]

    # data of the gold translation sentences for Bleu evaluation
    BLEU_GOLD_DATA = param_dict[Language(int(config['LANGUAGE']))]["BLEU_GOLD_DATA"]

    # the translations of the dataset sentences, using the debiased embedding table, with source line nums printed
    TRANSLATED_DEBIASED = NEMATUS_HOME + "en-" + lang + "/output/debiased_"+debias_method+".out.tmp"

    # the translations of the dataset sentences, using the non debiased embedding table, with source line nums printed
    TRANSLATED_NON_DEBIASED = NEMATUS_HOME + "en-" + lang + "/output/non_debiased_"+debias_method+".out.tmp"

    # # the translations of the dataset sentences, using the debiased embedding table, after filtering lines that are cummon with the non debiased translation
    # TRANSLATED_DEBIASED_MERGED =  NEMATUS_HOME + "en-" + lang + "/evaluate/debiased_"+debias_method+".txt"
    #
    # # the translations of the dataset sentences, using the non debiased embedding table, after filtering lines that are cummon with the debiased translation
    # TRANSLATED_NON_DEBIASED_MERGED = NEMATUS_HOME + "en-" + lang + "/evaluate/non_debiased_"+debias_method+".txt"

    # # the gold translations after filtering lines that were not translated
    # BLEU_GOLD_DATA_FILTERED = NEMATUS_HOME + "en-" + lang + "/evaluate/gold_"+debias_method+".txt"

    return BLEU_SOURCE_DATA, BLEU_GOLD_DATA, TRANSLATED_DEBIASED, TRANSLATED_NON_DEBIASED


#################debiaswe files#################
DEFINITIONAL_FILE = NEMATUS_HOME + "debiaswe/data/definitional_pairs.json"
GENDER_SPECIFIC_FILE = NEMATUS_HOME + "debiaswe/data/gender_specific_full.json"
PROFESSIONS_FILE = NEMATUS_HOME + "debiaswe/data/professions.json"
EQUALIZE_FILE = NEMATUS_HOME + "debiaswe/data/equalize_pairs.json"
