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




DEFINITIONAL_FILE = "/cs/usr/bareluz/gabi_labs/nematus/debiaswe/data/definitional_pairs.json"
GENDER_SPECIFIC_FILE = "/cs/usr/bareluz/gabi_labs/nematus/debiaswe/data/gender_specific_full.json"
EQUALIZE_FILE = "/cs/usr/bareluz/gabi_labs/nematus/debiaswe/data/equalize_pairs.json"
DEBIASED_TARGET_FILE = "/cs/usr/bareluz/gabi_labs/nematus/debiaswe/embeddings/Nematus-hard-debiased.bin"
EMBEDDING_TABLE_FILE = "/cs/labs/gabis/bareluz/nematus/embedding_table.bin"
EMBEDDING_DEBIASWE_FILE = "/cs/labs/gabis/bareluz/nematus/embedding_debiaswe.txt"
ENG_DICT_FILE = "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/train.clean.unesc.tok.tc.bpe.en.json"
np. set_printoptions(suppress=True)
output_translate_file = open('/cs/usr/bareluz/gabi_labs/nematus/output_translate.txt', 'r')
def check_all_lines_exist():
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
  return lines_count

def get_embedding_table():
  embedding_matrix = (np.zeros((29344,256))).astype(np.str)
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
              if lines_count[line_num]>0:
                continue
              lines_count[line_num]+=1
              row = i[i.find("[")+1:i.rfind("]")]
              row = row.split(" ")
              embedding_matrix[line_num,:]=row
  with open(EMBEDDING_TABLE_FILE, 'wb') as file_:
      pickle.dump(embedding_matrix,file_)
  return np.array(embedding_matrix,dtype=np.double)

def prepare_data_to_debias():
    with open(ENG_DICT_FILE, 'r') as dict_file, open(EMBEDDING_TABLE_FILE,'rb') as embedding_file, open(EMBEDDING_DEBIASWE_FILE, 'w') as embedding_debiaswe_file:
        eng_dictionary = json.load(dict_file)
        embeddings = pickle.load(embedding_file)
        eng_dictionary_list = list(eng_dictionary.keys())
        for i,w in enumerate(eng_dictionary_list):
            embedding_debiaswe_file.write(w+" "+' '.join(map(str, embeddings[i,:]))+"\n")

def debias_data():

    with open(DEFINITIONAL_FILE, "r") as f:
        defs = json.load(f)
    print("definitional", defs)

    with open(EQUALIZE_FILE, "r") as f:
        equalize_pairs = json.load(f)

    with open(GENDER_SPECIFIC_FILE, "r") as f:
        gender_specific_words = json.load(f)
    print("gender specific", len(gender_specific_words), gender_specific_words[:10])

    E = we.WordEmbedding(EMBEDDING_DEBIASWE_FILE)

    print("Debiasing...")
    debias(E, gender_specific_words, defs, equalize_pairs)

    print("Saving to file...")
    if EMBEDDING_DEBIASWE_FILE[-4:] == EMBEDDING_DEBIASWE_FILE[-4:] == ".bin":
        E.save_w2v(DEBIASED_TARGET_FILE)
    else:
        E.save(DEBIASED_TARGET_FILE)

def load_debiased():
    embedding_table = []
    with open(DEBIASED_TARGET_FILE, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.decode("utf-8")
            embedding = line.split(" ")[1:]
            embedding_table.append(embedding)
    return np.array(embedding_table)

if __name__ == '__main__':
  # embedding_matrix = get_embedding_table()
  # print(np.shape(embedding_matrix))
  # with open(EMBEDDING_TABLE_FILE, 'rb') as file_:
  #   embedding_matrix = pickle.load(file_)
  # print(embedding_matrix)

  # prepare_data_to_debias()
  # debias_data()
  embedding_table = load_debiased()
  print(np.shape(embedding_table))
  print(embedding_table)