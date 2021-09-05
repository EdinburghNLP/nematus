EN_ANTI_PARSED = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/anti.en"
EN_ANTI = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en_anti.txt"
RU_ANTI_DEBIASED = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/output/debiased_anti.out.tmp"
RU_ANTI_NON_DEBIASED = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/output/non_debiased_anti.out.tmp"
# RU_DEBIASED_EVAL = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/output/en-ru-debiased1.txt"
RU_DEBIASED_EVAL = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/mt_gender/translations/nematus/en-ru-debiased.txt"
# RU_NON_DEBIASED_EVAL = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-ru/output/en-ru-non-debiased1.txt"
RU_NON_DEBIASED_EVAL = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/mt_gender/translations/nematus//en-ru-non-debiased.txt"

def parse_gender_sents(source_filename, dest_filename):
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
    # parse_gender_sents(EN_ANTI, EN_ANTI_PARSED)
    prepare_gender_sents_translation_to_evaluation(EN_ANTI_PARSED,RU_ANTI_DEBIASED, RU_DEBIASED_EVAL)
    prepare_gender_sents_translation_to_evaluation(EN_ANTI_PARSED,RU_ANTI_NON_DEBIASED, RU_NON_DEBIASED_EVAL)