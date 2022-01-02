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

sys.path.append("..")  # Adds higher directory to python modules path.
from debiaswe.debiaswe import we
from debiaswe.debiaswe.debias import debias
import sys
from sklearn.decomposition import PCA
import sklearn
import random
from sklearn.svm import LinearSVC, SVC

sys.path.append("..")  # Adds higher directory to python modules path.
from consts import get_debias_files_from_config, EMBEDDING_SIZE, DEFINITIONAL_FILE, PROFESSIONS_FILE, \
    GENDER_SPECIFIC_FILE, EQUALIZE_FILE
sys.path.append("../..")  # Adds higher directory to python modules path.
from nullspace_projection.src.debias import load_word_vectors, project_on_gender_subspaces, get_vectors, get_debiasing_projection
np.set_printoptions(suppress=True)


class DebiasManager():

    def __init__(self, consts_config_str):
        self.E = None
        self.non_debiased_embeddings = None
        self.DICT_SIZE, self.ENG_DICT_FILE, self.OUTPUT_TRANSLATE_FILE, self.EMBEDDING_TABLE_FILE, \
        self.EMBEDDING_DEBIASWE_FILE, self.DEBIASED_TARGET_FILE = get_debias_files_from_config(consts_config_str)

    def __check_all_lines_exist(self):
        """
        checks that each line in the embedding table, printed in translate run, exists (since the lines are iterated with threads
        and are printed in random order)
        """
        lines_count = np.zeros(self.DICT_SIZE)
        with open(self.OUTPUT_TRANSLATE_FILE, "r") as output_translate_file:
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
        print("all lines exist?: " + str(not lines_count.__contains__(0)))
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
        embedding_matrix = (np.zeros((self.DICT_SIZE, EMBEDDING_SIZE))).astype(np.str)
        lines_count = np.zeros(self.DICT_SIZE)
        with open(self.OUTPUT_TRANSLATE_FILE, "r") as output_translate_file:
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
        self.non_debiased_embeddings = embedding_matrix
        return embedding_matrix

    def __prepare_data_to_debias(self, inlp=False):
        """
        given path to dictionary, the path to the embedding table saved in get_embedding_table() and the file name to save the data,
        it prepares the embedding table in the format of <word> <embedding>/n , this is the format that debias() in debiaswe, uses.
        saves the embedding with the desired format to self.EMBEDDING_DEBIASWE_FILE
        """

        with open(self.ENG_DICT_FILE, 'r') as dict_file, open(self.EMBEDDING_DEBIASWE_FILE, 'w') as dest_file:
            eng_dictionary = json.load(dict_file)
            if inlp:
                s = np.shape(self.non_debiased_embeddings)
                dest_file.write(str(s[0])+" "+str(s[1]) +"\n")
            for w, i in eng_dictionary.items():
                dest_file.write(w + " " + ' '.join(map(str, self.non_debiased_embeddings[i, :])) + "\n")

    def debias_inlp(self, by_pca):
        model, vecs, words = load_word_vectors(fname=self.EMBEDDING_DEBIASWE_FILE)
        num_vectors_per_class = 7500

        if by_pca:
            pairs = [("male", "female"), ("masculine", "feminine"), ("he", "she"), ("him", "her")]
            gender_vecs = [model[p[0]] - model[p[1]] for p in pairs]
            pca = PCA(n_components=1)
            pca.fit(gender_vecs)
            gender_direction = pca.components_[0]

        else:
            gender_direction = model["he"] - model["she"]
        gender_unit_vec = gender_direction / np.linalg.norm(gender_direction)
        masc_words_and_scores, fem_words_and_scores, neut_words_and_scores = project_on_gender_subspaces(
            gender_direction, model, n=num_vectors_per_class)
        masc_words, masc_scores = list(zip(*masc_words_and_scores))
        neut_words, neut_scores = list(zip(*neut_words_and_scores))
        fem_words, fem_scores = list(zip(*fem_words_and_scores))
        masc_vecs, fem_vecs = get_vectors(masc_words, model), get_vectors(fem_words, model)
        neut_vecs = get_vectors(neut_words, model)

        n = min(3000, num_vectors_per_class)
        all_significantly_biased_words = masc_words[:n] + fem_words[:n]
        all_significantly_biased_vecs = np.concatenate((masc_vecs[:n], fem_vecs[:n]))
        all_significantly_biased_labels = np.concatenate((np.ones(n, dtype=int),
                                                          np.zeros(n, dtype=int)))

        all_significantly_biased_words, all_significantly_biased_vecs, all_significantly_biased_labels = sklearn.utils.shuffle(
            all_significantly_biased_words, all_significantly_biased_vecs, all_significantly_biased_labels)
        # print(np.random.choice(masc_words, size = 75))
        print("TOP MASC")
        print(masc_words[:50])
        # print("LAST MASC")
        # print(masc_words[-120:])
        print("-------------------------")
        # print(np.random.choice(fem_words, size = 75))
        print("TOP FEM")
        print(fem_words[:50])
        # print("LAST FEM")
        # print(fem_words[-120:])
        print("-------------------------")
        # print(np.random.choice(neut_words, size = 75))
        print(neut_words[:50])

        print(masc_scores[:10])
        print(masc_scores[-10:])
        print(neut_scores[:10])

        random.seed(0)
        np.random.seed(0)

        X = np.concatenate((masc_vecs, fem_vecs, neut_vecs), axis=0)
        # X = (X - np.mean(X, axis = 0, keepdims = True)) / np.std(X, axis = 0)
        y_masc = np.ones(masc_vecs.shape[0], dtype=int)
        y_fem = np.zeros(fem_vecs.shape[0], dtype=int)
        y_neut = -np.ones(neut_vecs.shape[0], dtype=int)
        # y = np.concatenate((masc_scores, fem_scores, neut_scores))#np.concatenate((y_masc, y_fem))
        y = np.concatenate((y_masc, y_fem, y_neut))
        X_train_dev, X_test, y_train_dev, Y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3,
                                                                                            random_state=0)
        X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(X_train_dev, y_train_dev,
                                                                                  test_size=0.3,
                                                                                  random_state=0)
        print("Train size: {}; Dev size: {}; Test size: {}".format(X_train.shape[0], X_dev.shape[0], X_test.shape[0]))

        gender_clf = LinearSVC
        # gender_clf = SGDClassifier
        # gender_clf = LogisticRegression
        # gender_clf = LinearDiscriminantAnalysis
        # gender_clf = Perceptron

        params_svc = {'fit_intercept': False, 'class_weight': None, "dual": False, 'random_state': 0}
        params_sgd = {'fit_intercept': False, 'class_weight': None, 'max_iter': 1000, 'random_state': 0}
        params = params_svc
        # params = {'loss': 'hinge', 'n_jobs': 16, 'penalty': 'l2', 'max_iter': 2500, 'random_state': 0}
        # params = {}
        n = 35
        min_acc = 0
        is_autoregressive = True
        dropout_rate = 0

        P, rowspace_projs, Ws = get_debiasing_projection(gender_clf, params, n, 256, is_autoregressive, min_acc,
                                                                X_train, Y_train, X_dev, Y_dev,
                                                                Y_train_main=None, Y_dev_main=None,
                                                                by_class=False, dropout_rate=dropout_rate)
        return (P.dot(vecs.T)).T, gender_direction
    def __debias_bolukbasi(self, debiased_target_file=None):
        """
        debiases the nematus embedding table that was created through the learning phase and saved in prepare_data_to_debias()
        saves the
        """
        self.E = we.WordEmbedding(self.EMBEDDING_DEBIASWE_FILE)
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
        if (np.shape(embedding_table)[0] != self.DICT_SIZE):
            embedding_table = np.vstack([embedding_table, self.non_debiased_embeddings[-1]])
        return np.array(embedding_table).astype(np.float32)

    def __print_bias_amount(self, word, gender_direction, debiased_embedding, orig_embedding):
        if self.E is None:
            with open(self.ENG_DICT_FILE, 'r') as dict_file:
                index = json.load(dict_file)
        else:
            index = self.E.index
        if word in index:
            word_index = index[word]
            bieas_before = '{:.20f}'.format(np.dot(orig_embedding[word_index], gender_direction))
            bias_after = '{:.20f}'.format(np.dot(debiased_embedding[word_index], gender_direction))
            print(word + ": bias before debias= " + bieas_before + ". bias after debias= " + bias_after)

    def debias_sanity_check(self, embedding_table_file=None, debiased_embedding_table=None, gender_direction=None):
        print("*******************sanity check**************************")
        if embedding_table_file is None:
            embedding_table_file = self.EMBEDDING_TABLE_FILE
        if debiased_embedding_table is None:
            debiased_embedding = debiased_embedding_table
        else:
            debiased_embedding = self.load_debias_format_to_array(self.DEBIASED_TARGET_FILE)
        debiased_embedding = debiased_embedding.astype('float32')
        with open(DEFINITIONAL_FILE, "r") as f:
            defs = json.load(f)
        if gender_direction is None:
            if self.E is None:
                raise Exception("WordEmbedding E was not created")
            gender_direction = we.doPCA(defs, self.E).components_[0]
        with open(PROFESSIONS_FILE, "r") as f:
            professions = json.load(f)
        with open(embedding_table_file, 'rb') as embedding_file:
            orig_embedding = pickle.load(embedding_file)
            orig_embedding = orig_embedding.astype('float32')
        print("--------professions--------")
        for p in professions:
            self.__print_bias_amount(p[0], gender_direction, debiased_embedding, orig_embedding)

        with open(DEFINITIONAL_FILE, "r") as f:
            defs = json.load(f)
        print("--------gender specific--------")
        for a, b in defs:
            self.__print_bias_amount(a, gender_direction, debiased_embedding, orig_embedding)
            self.__print_bias_amount(b, gender_direction, debiased_embedding, orig_embedding)
        print("********************************************************")

    def load_and_debias(self,inlp):
        embedding_matrix = self.__get_non_debiased_embedding_table()
        self.__prepare_data_to_debias(inlp)
        # self.__debias_bolukbasi()
        # return self.load_debias_format_to_array()
        return self.debias_inlp(False)

if __name__ == '__main__':

    CONSTS_CONFIG_STR = "{'USE_DEBIASED': 0, 'LANGUAGE': 0, 'COLLECT_EMBEDDING_TABLE': 0, 'PRINT_LINE_NUMS': 0}"
    debias_manager = DebiasManager(CONSTS_CONFIG_STR)
    debiased_embedding, gender_direction = debias_manager.load_and_debias(inlp=True)

    print(np.shape(debiased_embedding))
    print(debiased_embedding)
    debias_manager.debias_sanity_check(debiased_embedding_table=debiased_embedding, gender_direction=gender_direction)
