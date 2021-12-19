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
print("in debias manager")
import numpy as np
import pickle
import json
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from debiaswe.debiaswe import we
from debiaswe.debiaswe.debias import debias
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from consts import get_debias_files_from_config, EMBEDDING_SIZE, DEFINITIONAL_FILE, PROFESSIONS_FILE, GENDER_SPECIFIC_FILE,EQUALIZE_FILE
# import numpy as np

np.set_printoptions(suppress=True)


class DebiasManager():

    def __init__(self, dict_size, eng_dict_file, output_translate_file, consts_config_str):
        """

        :param dict_size: the size of the english input dictionary
        :param eng_dict_file: the source english dictionary file path
        :param output_translate_file: the path of the file that translate wrote the embedding table to. this file will be parsed and debiased
        """
        self.output_translate_file = output_translate_file
        # self.E = we.WordEmbedding(EMBEDDING_DEBIASWE_FILE)
        self.E = None
        self.eng_dict_file = eng_dict_file
        self.dict_size = dict_size
        self.non_debiased_embeddings = None
        self.DICT_SIZE, self.ENG_DICT_FILE, self.OUTPUT_TRANSLATE_FILE, self.EMBEDDING_TABLE_FILE, \
        self.EMBEDDING_DEBIASWE_FILE, self.DEBIASED_TARGET_FILE = get_debias_files_from_config(consts_config_str)

    def __check_all_lines_exist(self):
        """
        checks that each line in the embedding table, printed in translate run, exists (since the lines are iterated with threads
        and are printed in random order)
        """
        lines_count = np.zeros(self.dict_size)
        with open(self.output_translate_file, "r") as output_translate_file:
            while True:
                line = output_translate_file.readline()
                if not line:
                    break
                if line.__contains__("enc_inputs for word"):
                    a = line.split("enc_inputs for word")
                    for i in a:
                        if i.__contains__("[") and not i.__contains__("embedding_table shape"):
                            line_num = i.split("[")[0]
                            lines_count[int(line_num)] += 1
        # for i in range(len(lines_count)):
        #     print("line num "+str(i)+": "+str(lines_count[i]))
        print("all lines exist?: "+str(not lines_count.__contains__(0)))
        return not lines_count.__contains__(0)

    def __get_non_debiased_embedding_table(self):
        """
        if the embedding table , printed in translate run, contains all lines, creates a matrix with the right order of
        lines of the embedding matrix learned during the train phase.
        then it saves the matrix to pickle and returns it
        :return:
        the embedding table as an numpy array
        """
        if not self.__check_all_lines_exist():
            raise Exception("not all lines exist in the embedding table")
        embedding_matrix = (np.zeros((self.dict_size, EMBEDDING_SIZE))).astype(np.str)
        lines_count = np.zeros(self.dict_size)
        with open(self.output_translate_file, "r") as output_translate_file:
            while True:
                line = output_translate_file.readline()
                if not line:
                    break
                if line.__contains__("enc_inputs for word"):
                    a = line.split("enc_inputs for word")
                    for i in a:
                        if i.__contains__("[") and not i.__contains__("embedding_table shape"):
                            line_num = int(i.split("[")[0])
                            if lines_count[line_num] > 0:
                                continue
                            lines_count[line_num] += 1
                            row = i[i.find("[") + 1:i.rfind("]")]
                            row = row.split(" ")
                            embedding_matrix[line_num, :] = row
        embedding_matrix = np.array(embedding_matrix, dtype=np.double)
        with open(self.EMBEDDING_TABLE_FILE, 'wb') as file_:
            pickle.dump(embedding_matrix, file_)
        return embedding_matrix

    def __prepare_data_to_debias(self, embedding_table_file=None, dest_file=None,
                                 non_debiased_embedding_table=None):
        """
        given path to dictionary, the path to the embedding table saved in get_embedding_table() and the file name to save the data,
        it prepares the embedding table in the format of <word> <embedding>/n , this is the format that debias() in debiaswe, uses.
        saves the embedding with the desired format to dest_file
        """
        if embedding_table_file is None:
            embedding_table_file = self.EMBEDDING_TABLE_FILE
        if dest_file is None:
            dest_file = self.EMBEDDING_DEBIASWE_FILE
        if non_debiased_embedding_table is None:
            with open(embedding_table_file, 'rb') as embedding_file:
                self.non_debiased_embeddings = pickle.load(embedding_file)
        else:
            self.non_debiased_embeddings = non_debiased_embedding_table
        with open(self.eng_dict_file, 'r') as dict_file, open(dest_file, 'w') as embedding_debiaswe_file:
            eng_dictionary = json.load(dict_file)
            # eng_dictionary_list = list(eng_dictionary.keys())
            # assert(list(eng_dictionary.values()) == list(range(self.dict_size)))
            # TODO: if this passes, then put a breakpoint here and examine the value of embeddings[-1, :], what's in there?
            for w, i in eng_dictionary.items():
                embedding_debiaswe_file.write(w + " " + ' '.join(map(str, self.non_debiased_embeddings[i, :])) + "\n")
        self.E = we.WordEmbedding(dest_file)

    def __debias_data(self, debiased_target_file=None):
        """
        debiases the nematus embedding table that was created through the learning phase and saved in prepare_data_to_debias()
        saves the
        """
        if debiased_target_file is None:
            debiased_target_file = self.DEBIASED_TARGET_FILE
        with open(DEFINITIONAL_FILE, "r") as f:
            defs = json.load(f)
        print("definitional", defs)

        with open(EQUALIZE_FILE, "r") as f:
            equalize_pairs = json.load(f)

        with open(GENDER_SPECIFIC_FILE, "r") as f:
            gender_specific_words = json.load(f)
        print("gender specific", len(gender_specific_words), gender_specific_words[:10])

        if self.E is None:
            raise Exception("WordEmbedding E was not created")
        print("Debiasing...")
        debias(self.E, gender_specific_words, defs, equalize_pairs)

        print("Saving to file...")
        if self.EMBEDDING_DEBIASWE_FILE[-4:] == self.EMBEDDING_DEBIASWE_FILE[-4:] == ".bin":
            self.E.save_w2v(debiased_target_file)
        else:
            self.E.save(debiased_target_file)

    def load_debias_format_to_array(self, filename=None):
        """
        loads a debiased embedding from filename and transforms it to numpy array
        :return: the debiased embedding table as numpy array
        """
        if filename is None:
            filename = self.DEBIASED_TARGET_FILE
        embedding_table = []
        with open(filename, "rb") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.decode("utf-8")
                embedding = line.split(" ")[1:]
                embedding_table.append(embedding)
        if (np.shape(embedding_table)[0]!=self.dict_size):
            embedding_table = np.vstack([embedding_table,self.non_debiased_embeddings[-1]])
        return np.array(embedding_table).astype(np.float32)

    def __print_bias_amount(self, word, gender_direction, debiased_embedding, orig_embedding):
        if self.E is None:
            raise Exception("WordEmbedding E was not created")
        if word in self.E.index:
            word_index = self.E.index[word]
            bieas_before = '{:.20f}'.format(np.dot(orig_embedding[word_index], gender_direction))
            bias_after = '{:.20f}'.format(np.dot(debiased_embedding[word_index], gender_direction))
            print(word + ": bias before debias= " + bieas_before + ". bias after debias= " + bias_after)

    def debias_sanity_check(self, embedding_table_file=None, debiased_embedding_table=None):
        if embedding_table_file is None:
            embedding_table_file = self.EMBEDDING_TABLE_FILE
        if self.E is None:
            raise Exception("WordEmbedding E was not created")
        print("*******************sanity check**************************")
        if debiased_embedding_table is None:
            debiased_embedding = debiased_embedding_table
        else:
            debiased_embedding = self.load_debias_format_to_array(self.DEBIASED_TARGET_FILE)
        debiased_embedding = debiased_embedding.astype('float32')
        with open(DEFINITIONAL_FILE, "r") as f:
            defs = json.load(f)
        gender_direction = we.doPCA(defs, self.E).components_[0]
        with open(PROFESSIONS_FILE, "r") as f:
            professions = json.load(f)
        with open(embedding_table_file, 'rb') as embedding_file:
            orig_embedding = pickle.load(embedding_file)
            orig_embedding = orig_embedding.astype('float32')
        for p in professions:
            self.__print_bias_amount(p[0], gender_direction, debiased_embedding, orig_embedding)

        with open(DEFINITIONAL_FILE, "r") as f:
            defs = json.load(f)
        for a, b in defs:
            self.__print_bias_amount(a, gender_direction, debiased_embedding, orig_embedding)
            self.__print_bias_amount(b, gender_direction, debiased_embedding, orig_embedding)
        print("********************************************************")
    def load_and_debias(self):
        embedding_matrix = self.__get_non_debiased_embedding_table()
        self.__prepare_data_to_debias(non_debiased_embedding_table=embedding_matrix)
        self.__debias_data()
        return self.load_debias_format_to_array()


# if __name__ == '__main__':
#     DICT_SIZE = 30545
#     ENG_DICT_FILE= "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/20.07.21//train.clean.unesc.tok.tc.bpe.en.json"
#     OUTPUT_TRANSLATE_FILE = "/cs/labs/gabis/bareluz/nematus_clean/nematus/en-he/debias/output_translate_he.txt"
#     CONSTS_CONFIG_STR = "{'USE_DEBIASED': 0, 'LANGUAGE': 2, 'COLLECT_EMBEDDING_TABLE': 1, 'PRINT_LINE_NUMS': 0}"
#     debias_manager = DebiasManager(DICT_SIZE,ENG_DICT_FILE,OUTPUT_TRANSLATE_FILE,CONSTS_CONFIG_STR)
#     # print("does all lines exist?: "+str(debias_manager.__check_all_lines_exist()))
#     debiased_embedding = debias_manager.load_and_debias()
#
#     print(np.shape(debiased_embedding))
#     print(debiased_embedding)
#     debias_manager.debias_sanity_check(debiased_embedding_table=debiased_embedding)
