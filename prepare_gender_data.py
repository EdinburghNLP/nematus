import sys
sys.path.append("..") # Adds higher directory to python modules path.
from consts import get_evaluate_gender_files
import argparse
def prepare_gender_sents_to_translation(source_filename, dest_filename):
    with open(source_filename, "r") as s, open(dest_filename, "w") as d:
        while True:
            line = s.readline()
            if not line:
                break
            d.write(line.split("\t")[2]+"\n")
def prepare_gender_sents_translation_to_evaluation(source_filename,source_translated_filename, dest_filename):
    with open(source_filename, "r") as s1,open(source_translated_filename, "r") as s2, open(dest_filename, "w") as d:
        while True:
            line_source = s1.readline()
            line_translated = s2.readline()
            if not line_source:
                break
            d.write(line_source[:-1]+" ||| "+line_translated[:-1]+"\n")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-c', '--config_file', type=str, required=True,
            help="path to a config file that conatains: \n"
                 "debiased= run translate on the debiased dictionary or not\n"
                 "language= the language to translate to from english. RUSSIAN = 0, GERMAN = 1, HEBREW = 2\n"
                 "collect_embedding_table= run translate to collect embedding table or not")
    args = parser.parse_args()
    EN_ANTI_PARSED, ANTI_TRANSLATED_DEBIASED, ANTI_TRANSLATED_NON_DEBIASED, \
    DEBIASED_EVAL, NON_DEBIASED_EVAL, EN_ANTI = get_evaluate_gender_files(args.config_file)

    # parse_gender_sents(EN_ANTI, EN_ANTI_PARSED)
    prepare_gender_sents_translation_to_evaluation(EN_ANTI_PARSED, ANTI_TRANSLATED_DEBIASED, DEBIASED_EVAL)
    prepare_gender_sents_translation_to_evaluation(EN_ANTI_PARSED, ANTI_TRANSLATED_NON_DEBIASED, NON_DEBIASED_EVAL)