# import tensorflow as tf
#
# embedding_table = tf.Variable(initial_value=None,name="embedding_table")
# # Add ops to save and restore all the variables.
# saver = tf.compat.v1.train.Saver({"embedding_table": embedding_table})
#
# # Later, launch the model, use the saver to restore variables from disk, and
# # do some work with the model.
# with tf.compat.v1.Session() as sess:
#   # Restore variables from disk.
#   saver.restore(sess, "/cs/labs/gabis/bareluz/nematus/output_translate.ckpt")
#   print(embedding_table)

import numpy as np
import pickle
import json
from debiaswe.debiaswe import we
from debiaswe.debiaswe.debias import debias
np.set_printoptions(suppress=True)



DEFINITIONAL_FILE = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/debiaswe/data/definitional_pairs.json"
GENDER_SPECIFIC_FILE = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/debiaswe/data/gender_specific_full.json"
PROFESSIONS_FILE = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/debiaswe/data/professions.json"
EQUALIZE_FILE = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/debiaswe/data/equalize_pairs.json"
DEBIASED_TARGET_FILE = "/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/debiaswe/embeddings/Nematus-hard-debiased.bin"
EMBEDDING_TABLE_FILE = "/cs/labs/gabis/bareluz/nematus_clean/nematus/embedding_table.bin"
EMBEDDING_DEBIASWE_FILE = "/cs/labs/gabis/bareluz/nematus_clean/nematus/embedding_debiaswe.txt"
ENG_DICT_FILE = "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/train.clean.unesc.tok.tc.bpe.en.json"
np. set_printoptions(suppress=True)
output_translate_file = open('/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/output_translate.txt', 'r')
E = we.WordEmbedding(EMBEDDING_DEBIASWE_FILE)

def check_all_lines_exist():
    """
    checks that each line in the embedding table, printed in translate run, exists (since the lines are iterated with threads
    and are printed in random order)
    """
    lines_count = np.zeros(29344)
    while True:
        line = output_translate_file.readline()
        if not line:
            break
        if line.__contains__("enc_inputs for word"):
            a = line.split("enc_inputs for word")
            for i in a:
                if i.__contains__("["):
                    line_num = i.split("[")[0]
                    lines_count[int(line_num)] += 1
    for i, l in enumerate(lines_count):
        print("entry num " + str(i) + ": " + str(l))
    return lines_count.__contains__(0)
    # return lines_count

def get_embedding_table():
    """
    if the embedding table , printed in translate run, contains all lines, creates a matrix with the right order of
    lines of the embedding matrix learned during the train phase.
    then it saves the matrix to pickle and returns it
    :return:
    the embedding table as an numpy array
    """
    if not check_all_lines_exist():
        print("not all lines exist in the embedding table")
        return
    embedding_matrix = (np.zeros((29344, 256))).astype(np.str)
    lines_count = np.zeros(29344)
    while True:
        line = output_translate_file.readline()
        if not line:
            break
        if line.__contains__("enc_inputs for word"):
            a = line.split("enc_inputs for word")
            for i in a:
                if i.__contains__("["):
                    line_num = int(i.split("[")[0])
                    if lines_count[line_num] > 0:
                        continue
                    lines_count[line_num] += 1
                    row = i[i.find("[") + 1:i.rfind("]")]
                    row = row.split(" ")
                    embedding_matrix[line_num, :] = row
    embedding_matrix = np.array(embedding_matrix, dtype=np.double)
    with open(EMBEDDING_TABLE_FILE, 'wb') as file_:
        pickle.dump(embedding_matrix, file_)
    return embedding_matrix

def prepare_data_to_debias(source_dict_file=ENG_DICT_FILE, embedding_table_file = EMBEDDING_TABLE_FILE, dest_file = EMBEDDING_DEBIASWE_FILE):
    """
    given path to dictionary, the path to the embedding table saved in get_embedding_table() and the file name to save the data,
    it prepares the embedding table in the format of <word> <embedding>/n , this is the format that debias() in debiaswe, uses.
    saves the embedding with the desired format to dest_file
    """
    with open(source_dict_file, 'r') as dict_file, open(embedding_table_file,'rb') as embedding_file, open(dest_file, 'w') as embedding_debiaswe_file:
        eng_dictionary = json.load(dict_file)
        embeddings = pickle.load(embedding_file)
        eng_dictionary_list = list(eng_dictionary.keys())
        for i,w in enumerate(eng_dictionary_list):
            embedding_debiaswe_file.write(w+" "+' '.join(map(str, embeddings[i,:]))+"\n")
def debias_data(debiased_target_file=DEBIASED_TARGET_FILE):
    """
    debiases the nematus embedding table that was created through the learning phase and saved in prepare_data_to_debias()
    saves the
    """
    with open(DEFINITIONAL_FILE, "r") as f:
        defs = json.load(f)
    print("definitional", defs)

    with open(EQUALIZE_FILE, "r") as f:
        equalize_pairs = json.load(f)

    with open(GENDER_SPECIFIC_FILE, "r") as f:
        gender_specific_words = json.load(f)
    print("gender specific", len(gender_specific_words), gender_specific_words[:10])


    print("Debiasing...")
    debias(E, gender_specific_words, defs, equalize_pairs)

    print("Saving to file...")
    if EMBEDDING_DEBIASWE_FILE[-4:] == EMBEDDING_DEBIASWE_FILE[-4:] == ".bin":
        E.save_w2v(debiased_target_file)
    else:
        E.save(debiased_target_file)

def load_debias_format_to_array(filename=DEBIASED_TARGET_FILE):
    """
    loads a debiased embedding from filename and transforms it to numpy array
    :return: the debiased embedding table as numpy array
    """
    embedding_table = []
    with open(filename, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.decode("utf-8")
            embedding = line.split(" ")[1:]
            embedding_table.append(embedding)
    return np.array(embedding_table)

def print_bias_amount(word,gender_direction, debiased_embedding, orig_embedding):
    if word in E.index:
        word_index = E.index[word]
        bieas_before = '{:.20f}'.format(np.dot(orig_embedding[word_index], gender_direction))
        bias_after = '{:.20f}'.format(np.dot(debiased_embedding[word_index], gender_direction))
        print(word + ": bias before debias= " + bieas_before + ". bias after debias= " + bias_after)

def debias_sanity_check(embedding_table_file = EMBEDDING_TABLE_FILE, debiased_embedding_table=None):
    if debiased_embedding_table is not None:
        debiased_embedding = debiased_embedding_table
    else:
        debiased_embedding = load_debias_format_to_array(DEBIASED_TARGET_FILE)
    debiased_embedding = debiased_embedding.astype('float32')
    with open(DEFINITIONAL_FILE, "r") as f:
        defs = json.load(f)
    gender_direction = we.doPCA(defs, E).components_[0]
    with open(PROFESSIONS_FILE, "r") as f:
        professions = json.load(f)
    with open(embedding_table_file,'rb') as embedding_file:
        orig_embedding = pickle.load(embedding_file)
        orig_embedding= orig_embedding.astype('float32')
    for p in professions:
        print_bias_amount(p[0], gender_direction, debiased_embedding, orig_embedding)

    with open(DEFINITIONAL_FILE, "r") as f:
        defs = json.load(f)
    for a, b in defs:
        print_bias_amount(a, gender_direction, debiased_embedding, orig_embedding)
        print_bias_amount(b, gender_direction, debiased_embedding, orig_embedding)


if __name__ == '__main__':
    # print("does all lines exist?: "+str(check_all_lines_exist()))
    # embedding_matrix = get_embedding_table()
    # print(np.shape(embedding_matrix))
    # with open(EMBEDDING_TABLE_FILE, 'rb') as file_:
    #   embedding_matrix = pickle.load(file_)
    # print(embedding_matrix)

    # prepare_data_to_debias()
    # debias_data()
    # embedding_table = load_debias_format_to_array()
    # print(np.shape(embedding_table))
    # print(embedding_table)
    debias_sanity_check()