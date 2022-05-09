from easynmt import EasyNMT
from sacrebleu.metrics import BLEU
from detokenize import detokenize_matrix
import sys
sys.path.append("../../") # Adds higher directory to python modules path.
from consts import get_debias_files_from_config, EMBEDDING_SIZE, DEFINITIONAL_FILE, PROFESSIONS_FILE, \
    GENDER_SPECIFIC_FILE, EQUALIZE_FILE, get_basic_configurations, DebiasMethod
model = EasyNMT('opus-mt')
GOLD_HOME = "/cs/snapless/oabend/borgr/SSMT/data/"
ru_data = GOLD_HOME + "en_ru/newstest2019-enru.en"
de_data = GOLD_HOME + "en_de/newstest2012.en"
he_data = GOLD_HOME + "en_he/dev.en"

ru_gold = GOLD_HOME + "en_ru/newstest2019-enru.ru"
de_gold = GOLD_HOME + "en_de/newstest2012.de"
he_gold = GOLD_HOME + "en_he/dev.he"
def check_easynmt():

    # #Translate several sentences to German
    # sentences = ['You can define a list with sentences.',
    #              'All sentences are translated to your target language.',
    #              'Note, you could also mix the languages of the sentences.']
    bleu = BLEU()
    with open(ru_data, 'r') as ru, open(de_data, 'r') as de, open(he_data, 'r') as he, \
        open(ru_gold, 'r') as ru_translation_gold, open(de_gold, 'r') as de_translation_gold, open(he_gold, 'r') as he_translation_gold:

        print("translating ru")
        ru_translation = model.translate(ru.readlines(), source_lang='en', target_lang='ru', show_progress_bar=True)
        print("ru")
        print(bleu.corpus_score(detokenize_matrix(ru_translation,'ru'), [detokenize_matrix(ru_translation_gold.readlines(), 'ru')]))
        print("translating de")
        de_translation = model.translate(de.readlines(), source_lang='en', target_lang='de', show_progress_bar=True)
        print("de")
        print(bleu.corpus_score(detokenize_matrix(de_translation,'de'), [detokenize_matrix(de_translation_gold.readlines(),'de')]))
        print("translating he")
        he_translation = model.translate(he.readlines(), source_lang='en', target_lang='he', show_progress_bar=True)
        print("he")
        print(bleu.corpus_score(detokenize_matrix(he_translation,'he'), [detokenize_matrix(he_translation_gold.readlines(),'he')]))

# def debias_sanity_check(self, debiased_embedding_table=None):
#     """
#     prints the bias amount before and after debias for words that describes profession, and gender specific words
#     the biases are also printed to a csv file
#     :param debiased_embedding_table:
#     :return:
#     """
#     print("*******************sanity check**************************")
#     if debiased_embedding_table is None:
#         debiased_embedding_table = self.load_debias_format_to_array(self.DEBIASED_EMBEDDING)
#     debiased_embedding_table = debiased_embedding_table.astype('float32')
#     gender_direction = self.get_gender_direction()
#     with open(PROFESSIONS_FILE, "r") as f:
#         professions = json.load(f)
#     with open(self.EMBEDDING_TABLE_FILE, 'rb') as embedding_file:
#         orig_embedding = pickle.load(embedding_file)
#         orig_embedding = orig_embedding.astype('float32')
#     with open(self.SANITY_CHECK__FILE, 'wt') as f:
#         writer = csv.writer(f)
#         writer.writerow(["word", "bias before", "bias after"])
#     print("--------professions--------")
#     for p in professions:
#         self.__print_bias_amount(p[0], gender_direction, debiased_embedding_table, orig_embedding)
#
#     with open(DEFINITIONAL_FILE, "r") as f:
#         defs = json.load(f)
#     print("--------gender specific--------")
#     for a, b in defs:
#         self.__print_bias_amount(a, gender_direction, debiased_embedding_table, orig_embedding)
#         self.__print_bias_amount(b, gender_direction, debiased_embedding_table, orig_embedding)
#     print("********************************************************")
if __name__ == '__main__':
    check_easynmt()