import argparse
import re
from consts import LANGUAGE_STR_MAP,DebiasMethod, DEBIAS_MANAGER_HOME
import pandas as pd
import numpy as np
import json
import csv
import math
from datetime import datetime

LANGUAGES =LANGUAGE_STR_MAP.values()
DEBIAS_METHODS = [d.value for d in DebiasMethod]

def get_translation_results(file_name):
    result = {}
    with open(file_name) as f:
        lines = f.readlines()
        debiased_line = lines.index("debiased\n")+1
        result["debiased"] = float(lines[debiased_line].split(" ")[2])
        non_debiased_line = lines.index("non debiased\n") + 1
        result["non_debiased"] = float(lines[non_debiased_line].split(" ")[2])
    return result
def get_gender_results(file_name):
    result = {}
    with open(file_name) as f:
        line = f.readline()
        line_num=1
        debiased_found = False
        non_debiased_found = False

        while line:
            if line.__contains__("*debiased results*"):
                debiased_found = True
            if line.__contains__("*non debiased results*"):
                non_debiased_found = True
            match = re.search("{\"acc\": (.*), \"f1_male\": .*, \"f1_female\": .*, \"unk_male\": .*, \"unk_female\": .*, \"unk_neutral\": .*}",line)
            if match:
                accuracy = match.group(1)
                if debiased_found:
                    result["debiased"] = float(accuracy)
                    debiased_found = False
                elif non_debiased_found:
                    result["non_debiased"] = float(accuracy)
                    non_debiased_found =False
            line = f.readline()
            line_num+=1
    return result

def get_all_results(files_dict):

    results = {}
    for language in LANGUAGES:
        results[language] = {}
        for debias_method in DEBIAS_METHODS:
            results[language][debias_method] = {}
            results[language][debias_method]["translation"] = get_translation_results(result_files[language][debias_method]["translation"])
            results[language][debias_method]["gender"] = get_gender_results(result_files[language][debias_method]["gender"])
    print(json.dumps(results, sort_keys=True, indent=4))
    methods_results = []
    for debias_method in DEBIAS_METHODS:
        orig_results = []
        method_results = []
        for lang in LANGUAGES:
            orig_bleu = results[lang][0]["translation"]["non_debiased"]
            orig_delta_s = results[lang][0]["gender"]["non_debiased"]
            orig_results += [orig_bleu, -math.inf, orig_delta_s, -math.inf]
            method_bleu = results[lang][debias_method]["translation"]["debiased"]
            method_delta_s = results[lang][debias_method]["gender"]["debiased"]

            method_results += [method_bleu, method_bleu-orig_bleu, method_delta_s, method_delta_s-orig_delta_s]
        methods_results+=[method_results]
    results =np.around([orig_results, methods_results[0], methods_results[1]], 2)
    results[results==-math.inf] = None
    return results

def write_results_to_csv(results):
    headers = [None, "Russian", None, None, None, "German",None, None, None,  "Hebrew", None, None, None]
    sub_headers = [None]+["Bleu", "delta Bleu", "delta s", "delta delta s"]*3
    index = [["Original"], ["Bolukbasy"], ["Null It Out"]]
    data = np.append(index, results, axis=1)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    with open(DEBIAS_MANAGER_HOME+"results/results_"+dt_string+".csv", 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(sub_headers)
        writer.writerows(data)
def write_results_to_table(results):


    iterables = [["Russian", "German",  "Hebrew"], ["Bleu", "delta Bleu", "delta s", "delta delta s"]]
    index =pd.MultiIndex.from_product(iterables)
    df = pd.DataFrame(results, index=["Original", "Bolukbasy", "Null It Out"], columns=index)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    with open(NEMATUS_HOME+"results/results_"+dt_string+".tex", 'w') as f:
        f.write(df.to_latex())
    pass
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #         '-c', '--config_str', type=str, required=True,
    #         help="a config dictionary str that conatains: \n"
    #              "debiased= run translate on the debiased dictionary or not\n"
    #              "language= the language to translate to from english. RUSSIAN = 0, GERMAN = 1, HEBREW = 2\n"
    #              "collect_embedding_table= run translate to collect embedding table or not\n"
    #              "print_line_nums= whether to print line numbers to output file in translate")
    # args = parser.parse_args()
    # TRANSLATION_EVALUATION, GENDER_EVALUATION = get_results_files(args.config_str)

    result_files = {}
    for language in LANGUAGES:
        result_files[language] = {}
        for debias_method in DEBIAS_METHODS:
            result_files[language][debias_method] = {}
            result_files[language][debias_method]["translation"] = DEBIAS_MANAGER_HOME + "en-" + language + "/debias/translation_evaluation_" + language + "_"+str(debias_method)+".txt"
            result_files[language][debias_method]["gender"] = DEBIAS_MANAGER_HOME + "en-" + language + "/debias/gender_evaluation_" + language + "_"+str(debias_method)+".txt"
    res = get_all_results(result_files)
    for i in res:
        print(i)
    # write_results_to_table(res)
    write_results_to_csv(res)