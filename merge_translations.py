import pickle
from typing import List
import re
import argparse
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from consts import get_evaluate_translation_files, get_evaluate_gender_files


def merge_two_translations(trans_file1: str, trans_file2: str, dest_file1: str, dest_file2: str, src_gold: str, dst_gold: str):
    """
    given debiased and non debiased translations in the form of <line num> <translation>, and a gold translations,
    keeps only the line numbers that are common to the debiased and non debiased translations.
    :param trans_file1: first translation file path (debiased)
    :param trans_file2: second translation file path (non debiased, the order doesn't matter)
    :param dest_file1: path to keep the filtered sentences from trans_file1
    :param dest_file2: path to keep the filtered sentences from trans_file2
    :param src_gold: gold translation file path
    :param dst_gold: path to keep the filtered sentences from src_gold
    """
    with open(trans_file1,"r") as f:
        lines1 = f.readlines()
    with open(trans_file2,"r") as f:
        lines2 = f.readlines()
    with open(src_gold, "r") as f:
        gold_lines =  f.readlines()
    line_nums1,line_nums2 = [line.split(" ||| ")[0] for line in lines1], [line.split(" ||| ")[0]for line in lines2]
    lines_to_keep = list(set(line_nums1) & set(line_nums2))

    lines_to_keep1 = []
    lines_to_keep2 = []
    lines_to_keep_gold = []

    for line_idx in range(len(lines1)):
        if line_nums1[line_idx] in lines_to_keep:
            lines_to_keep1.append(lines1[line_idx].split(" ||| ")[1])
    with open(dest_file1, "wb") as f:
        pickle.dump(lines_to_keep1,f)

    for line_idx in range(len(lines2)):
        if line_nums2[line_idx] in lines_to_keep:
            lines_to_keep2.append(lines2[line_idx].split(" ||| ")[1])
    with open(dest_file2, "wb") as f:
        pickle.dump(lines_to_keep2,f)

    with open(dst_gold, "w") as f:
        for i in range(len(gold_lines)):
            if str(i + 1) in lines_to_keep:
                f.write(gold_lines[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-c', '--config_str', type=str, required=True,
            help="a config dictionary str that conatains: \n"
                 "debiased= run translate on the debiased dictionary or not\n"
                 "language= the language to translate to from english. RUSSIAN = 0, GERMAN = 1, HEBREW = 2\n"
                 "collect_embedding_table= run translate to collect embedding table or not\n"
                 "print_line_nums= whether to print line numbers to output file in translate")
    parser.add_argument(
            '-e', '--eval_type', type=int, required=True, choices=[0,1],
            help="if 0 given, prepares data for gender evaluation.\n"
                 "if 1 given, prepares data for translation evaluation")
    args = parser.parse_args()

    if args.eval_type == 0:
        ANTI_TRANSLATED_DEBIASED, ANTI_TRANSLATED_NON_DEBIASED, DEBIASED_EVAL, NON_DEBIASED_EVAL, \
        EN_ANTI_MERGED, ANTI_TRANSLATED_DEBIASED_PICKLE, \
        ANTI_TRANSLATED_NON_DEBIASED_PICKLE, EN_ANTI_MT_GENDER = get_evaluate_gender_files(args.config_str)

        merge_two_translations(ANTI_TRANSLATED_DEBIASED, ANTI_TRANSLATED_NON_DEBIASED, ANTI_TRANSLATED_DEBIASED_PICKLE,
                               ANTI_TRANSLATED_NON_DEBIASED_PICKLE, EN_ANTI_MT_GENDER, EN_ANTI_MERGED)
    else:
        BLEU_SOURCE_DATA, BLEU_GOLD_DATA, TRANSLATED_DEBIASED, TRANSLATED_NON_DEBIASED, TRANSLATED_DEBIASED_PICKLE, \
        TRANSLATED_NON_DEBIASED_PICKLE, BLEU_GOLD_DATA_FILTERED = get_evaluate_translation_files(args.config_str)

        merge_two_translations(TRANSLATED_DEBIASED, TRANSLATED_NON_DEBIASED,TRANSLATED_DEBIASED_PICKLE,
                               TRANSLATED_NON_DEBIASED_PICKLE,BLEU_GOLD_DATA,BLEU_GOLD_DATA_FILTERED)

