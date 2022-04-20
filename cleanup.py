import os
from os import listdir
from os.path import isfile, join
import shutil

def cleanup(paths,files_to_ignore):
    for path in paths:
        dst_path = path +"/backup"
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        files = [f for f in listdir(path) if isfile(join(path, f)) and f not in files_to_ignore]
        for file in files:
            shutil.move(os.path.join(path, file), os.path.join(dst_path, file))

if __name__ == '__main__':
    languages = ["de", "ru", "he"]
    # / cs / usr / bareluz / gabi_labs / nematus_clean / nematus / en - < lang > / debias
    # / cs / usr / bareluz / gabi_labs / nematus_clean / nematus / en - < lang > / evaluate
    # / cs / usr / bareluz / gabi_labs / nematus_clean / nematus / en - < lang > / output
    # / cs / usr / bareluz / gabi_labs / nematus_clean / mt_gender / translations / nematus
    # / cs / usr / bareluz / gabi_labs / nematus_clean / mt_gender / data / aggregates

    for language in languages:
        cleanup(["/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-"+language +"/debias",
                 "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-"+language +"/evaluate",
                 "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-"+language +"/output",],
                ["output_translate_"+language +".txt"])
    cleanup(["/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/translations/nematus",
             "/cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender/data/aggregates"],
            ["en_anti.txt"])

