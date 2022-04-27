import os
from os import listdir
from os.path import isfile, join
import shutil
import argparse

def cleanup(paths,files_to_ignore):
    for path in paths:
        dst_path = path +"/backup"
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        files = [f for f in listdir(path) if isfile(join(path, f)) and f not in files_to_ignore]
        for file in files:
            shutil.move(os.path.join(path, file), os.path.join(dst_path, file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--clean_embedding_table', action='store_true',
        help="weather should clean the embedding table or not")
    parser.add_argument(
        '-t', '--clean_translation_files', action='store_true',
        help="weather should clean the translation files")
    args = parser.parse_args()
    # / cs / labs / gabis / bareluz / nematus_clean / nematus / en - ru / output / debiased_anti_0.out.tmp
    # / cs / labs / gabis / bareluz / nematus_clean / nematus / en - ru / output / non_debiased_anti_0.out.tmp
    # / cs / labs / gabis / bareluz / nematus_clean / nematus / en - ru / output / debiased_0.out.tmp
    # / cs / labs / gabis / bareluz / nematus_clean / nematus / en - ru / output / non_debiased_0.out.tmp
    # / cs / labs / gabis / bareluz / nematus_clean / nematus / en - ru / output / debiased_anti_1.out.tmp
    # / cs / labs / gabis / bareluz / nematus_clean / nematus / en - ru / output / non_debiased_anti_1.out.tmp
    # / cs / labs / gabis / bareluz / nematus_clean / nematus / en - ru / output / debiased_1.out.tmp
    # / cs / labs / gabis / bareluz / nematus_clean / nematus / en - ru / output / non_debiased_1.out.tmp
    languages = ["de","ru","he"]
    for language in languages:
        files_to_ignore = []
        if not args.clean_embedding_table:
            files_to_ignore.append("output_translate_" + language + ".txt")

        if not args.clean_translation_files:
            files_to_ignore+=["debiased_anti_0.out.tmp",
                              "non_debiased_anti_0.out.tmp",
                              "debiased_0.out.tmp",
                              "non_debiased_0.out.tmp",
                              "debiased_anti_1.out.tmp",
                              "non_debiased_anti_1.out.tmp",
                              "debiased_1.out.tmp",
                              "non_debiased_1.out.tmp",
                              ]

        cleanup(["/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-"+language +"/debias",
                 "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-"+language +"/evaluate",
                 "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-"+language +"/output",],
                files_to_ignore)
    cleanup(["/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/translations/nematus",
             "/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/data/aggregates"],
            ["en_anti.txt"])

