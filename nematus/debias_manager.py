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
import csv

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
    GENDER_SPECIFIC_FILE, EQUALIZE_FILE, get_basic_configurations, DebiasMethod

sys.path.append("../..")  # Adds higher directory to python modules path.
from nullspace_projection.src.debias import load_word_vectors, project_on_gender_subspaces, get_vectors, \
    get_debiasing_projection
from nullspace_projection.notebooks.inlp_functions import tsne, compute_v_measure

np.set_printoptions(suppress=True)


class DebiasManager():

    def __init__(self, consts_config_str):
        self.consts_config_str = consts_config_str
        self.DICT_SIZE, self.ENG_DICT_FILE, self.OUTPUT_TRANSLATE_FILE, self.EMBEDDING_TABLE_FILE, \
        self.EMBEDDING_DEBIASWE_FILE, self.DEBIASED_EMBEDDING, self.SANITY_CHECK__FILE = get_debias_files_from_config(
            consts_config_str)
        self.non_debiased_embeddings = self.get_non_debiased_embedding_table()

    @staticmethod
    def get_manager_instance(consts_config_str):
        """
        gets a configuration dictionary of the current run, and extracts the debias method from it.
        according to the debias method returns the relevant debias manager instance.
        :param consts_config_str: a configuration dictionary of the current
        :return: instance of a relevant debias manager
        """
        _, _, _, DEBIAS_METHOD = get_basic_configurations(consts_config_str)
        if DebiasMethod(DEBIAS_METHOD) == DebiasMethod.BOLUKBASY:
            return DebiasBlukbasyManager(consts_config_str)
        elif DebiasMethod(DEBIAS_METHOD) == DebiasMethod.NULL_IT_OUT:
            return DebiasINLPManager(consts_config_str)
        else:
            raise Exception("the debias method is incorrect")

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

    def get_non_debiased_embedding_table(self):
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
        embedding_matrix = embedding_matrix.astype(np.double)
        with open(self.EMBEDDING_TABLE_FILE, 'wb') as file_:
            pickle.dump(embedding_matrix, file_)
        return embedding_matrix

    def compare_tsne(self, X_dev, Y_dev, X_train, Y_train, X_test, Y_test, all_significantly_biased_vecs,
                     all_significantly_biased_labels, n, P, rowspace_projs, input_dim):
        X_dev = X_dev[Y_dev != -1]
        X_train = X_train[Y_train != -1]
        X_test = X_test[Y_test != -1]

        Y_dev = Y_dev[Y_dev != -1]
        Y_train = Y_train[Y_train != -1]
        Y_test = Y_test[Y_test != -1]

        M = 2000
        ind2label = {1: "Male-biased", 0: "Female-biased"}
        tsne_before = tsne(all_significantly_biased_vecs[:M], all_significantly_biased_labels[:M],
                           title="Original (t=0)",
                           ind2label=ind2label)
        X_dev_cleaned = (P.dot(X_dev.T)).T
        X_test_cleaned = (P.dot(X_test.T)).T
        X_trained_cleaned = (P.dot(X_train.T)).T
        all_significantly_biased_cleaned = P.dot(all_significantly_biased_vecs.T).T

        tsne_after = tsne(all_significantly_biased_cleaned[:M], all_significantly_biased_labels[:M],
                          title="Projected (t={})".format(n), ind2label=ind2label)

        print("V-measure-before (TSNE space): {}".format(
            compute_v_measure(tsne_before, all_significantly_biased_labels[:M])))
        print("V-measure-after (TSNE space): {}".format(
            compute_v_measure(tsne_after, all_significantly_biased_labels[:M])))

        print("V-measure-before (original space): {}".format(
            compute_v_measure(all_significantly_biased_vecs[:M], all_significantly_biased_labels[:M]), k=2))
        print("V-measure-after (original space): {}".format(compute_v_measure(X_test_cleaned[:M], Y_test[:M]), k=2))

        rank_before = np.linalg.matrix_rank(X_train)
        rank_after = np.linalg.matrix_rank(X_trained_cleaned)
        print("Rank before: {}; Rank after: {}".format(rank_before, rank_after))

        for t in [1, 3, 12, 18, 25, 35]:
            p = debias.get_projection_to_intersection_of_nullspaces(rowspace_projs[:t], input_dim)
            tsne_after = tsne(p.dot(all_significantly_biased_vecs[:M].T).T, all_significantly_biased_labels[:M],
                              title="Projected (t={})".format(t), ind2label=ind2label)

    def load_debias_format_to_array(self, filename):
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
        if (np.shape(embedding_table)[0] != self.DICT_SIZE):
            embedding_table = np.vstack([embedding_table, self.non_debiased_embeddings[-1]])
        return np.array(embedding_table).astype(np.float32)

    def __print_bias_amount(self, word, gender_direction, debiased_embedding, orig_embedding):
        """
        prints the bias of a word according to the gender direction before and after debias.
        the bias amount is in the range of -1 to 1 where 0 is neutral and the edges are male and female bias.
        the biases are also printed to a csv file
        :param word: embedding of a word to check
        :param gender_direction: the gender direction that determines the bias amount
        :param debiased_embedding: debiased embedding table as numpy array
        :param orig_embedding: non debiased embedding table as numpy array
        """
        with open(self.ENG_DICT_FILE, 'r') as dict_file:
            index = json.load(dict_file)
        # if self.E is None:
        #     pass
        # else:
        #     index = self.E.index
        f = open(self.SANITY_CHECK__FILE, 'a')
        writer = csv.writer(f)

        if word in index:
            word_index = index[word]
            bias_before = np.dot(orig_embedding[word_index], gender_direction)
            bias_after = np.dot(debiased_embedding[word_index], gender_direction)
            writer.writerow([word, bias_before, bias_after])
            bieas_before = '{:.20f}'.format(bias_before)
            bias_after = '{:.20f}'.format(bias_after)
            print(word + ": bias before debias= " + bieas_before + ". bias after debias= " + bias_after)
        f.close()

    def debias_sanity_check(self, debiased_embedding_table=None):
        """
        prints the bias amount before and after debias for words that describes profession, and gender specific words
        the biases are also printed to a csv file
        :param debiased_embedding_table:
        :return:
        """
        print("*******************sanity check**************************")
        if debiased_embedding_table is None:
            debiased_embedding_table = self.load_debias_format_to_array(self.DEBIASED_EMBEDDING)
        debiased_embedding_table = debiased_embedding_table.astype('float32')
        gender_direction = self.get_gender_direction()
        with open(PROFESSIONS_FILE, "r") as f:
            professions = json.load(f)
        with open(self.EMBEDDING_TABLE_FILE, 'rb') as embedding_file:
            orig_embedding = pickle.load(embedding_file)
            orig_embedding = orig_embedding.astype('float32')
        with open(self.SANITY_CHECK__FILE, 'wt') as f:
            writer = csv.writer(f)
            writer.writerow(["word", "bias before", "bias after"])
        print("--------professions--------")
        for p in professions:
            self.__print_bias_amount(p[0], gender_direction, debiased_embedding_table, orig_embedding)

        with open(DEFINITIONAL_FILE, "r") as f:
            defs = json.load(f)
        print("--------gender specific--------")
        for a, b in defs:
            self.__print_bias_amount(a, gender_direction, debiased_embedding_table, orig_embedding)
            self.__print_bias_amount(b, gender_direction, debiased_embedding_table, orig_embedding)
        print("********************************************************")

    def get_gender_direction(self):
        raise NotImplementedError()

    def __prepare_data_to_debias(self):
        """
        given path to dictionary, the path to the embedding table saved in get_embedding_table() and the file name to save the data,
        it prepares the embedding table in the format of <word> <embedding>/n , this is the format that debias() in debiaswe, uses.
        saves the embedding with the desired format to self.EMBEDDING_DEBIASWE_FILE
        """
        raise NotImplementedError()

    def debias_embedding_table(self):
        """
        debiases the embedding table according to the chosen method
        :return: the debiased embedding table
        """
        raise NotImplementedError()

    def save(self, debiased_embeddings):
        with open(self.ENG_DICT_FILE, 'r') as dict_file, open(self.DEBIASED_EMBEDDING, "w") as f:
            eng_dictionary = json.load(dict_file)
            for w, i in eng_dictionary.items():
                f.write(w + " " + ' '.join(map(str, debiased_embeddings[i, :])) + "\n")
        # with open(filename, "w") as f:
        #     f.write("\n".join([w+" " + " ".join([str(x) for x in v]) for w, v in zip(self.words, self.vecs)]))
        print("Wrote debiased embeddings to", self.DEBIASED_EMBEDDING)

class DebiasINLPManager(DebiasManager):
    def __init__(self, consts_config_str):
        super().__init__(consts_config_str)
        self.__prepare_data_to_debias()
        self.by_pca = False

    def __prepare_data_to_debias(self):
        """
        given path to dictionary, the path to the embedding table saved in get_embedding_table() and the file name to save the data,
        it prepares the embedding table in the format of <word> <embedding>/n , this is the format that debias() in debiaswe, uses.
        saves the embedding with the desired format to self.EMBEDDING_DEBIASWE_FILE
        """
        with open(self.ENG_DICT_FILE, 'r') as dict_file, open(self.EMBEDDING_DEBIASWE_FILE, 'w') as dest_file:
            eng_dictionary = json.load(dict_file)
            s = np.shape(self.non_debiased_embeddings)
            dest_file.write(str(s[0]) + " " + str(s[1]) + "\n")
            for w, i in eng_dictionary.items():
                dest_file.write(w + " " + ' '.join(map(str, self.non_debiased_embeddings[i, :])) + "\n")

    def get_gender_direction(self):
        """
        :return: a vector represents the gender direction
        """
        if self.by_pca:
            pairs = [("woman", "man"), ("girl", "boy"), ("she", "he"), ("mother", "father"),
                     ("daughter", "son"), ("gal", "guy"), ("female", "male"), ("her", "his"),
                     ("herself", "himself"), ("Mary", "John")]
            # pairs = [("male", "female"), ("masculine", "feminine"), ("he", "she"), ("him", "her")]
            gender_vecs = [self.model[p[0]] - self.model[p[1]] for p in pairs]
            pca = PCA(n_components=1)
            pca.fit(gender_vecs)
            gender_direction = pca.components_[0]

        else:
            gender_direction = self.model["he"] - self.model["she"]
        return gender_direction

    def debias_inlp_preparation(self):
        """
        prepares the classifier model, and the embeddings with their classification as male, female and neutral
        and separated to train dev and test
        :return:
        X_dev, Y_dev, X_train, Y_train, X_test, Y_test: the splitted embeddings and their tags
        all_significantly_biased_vecs, all_significantly_biased_labels: a set of significally biased embeddings
        vecs: the embeddings
        """
        self.model, vecs, words = load_word_vectors(fname=self.EMBEDDING_DEBIASWE_FILE)
        num_vectors_per_class = 7500
        gender_direction = self.get_gender_direction()

        gender_unit_vec = gender_direction / np.linalg.norm(gender_direction)
        masc_words_and_scores, fem_words_and_scores, neut_words_and_scores = project_on_gender_subspaces(
            gender_direction, self.model, n=num_vectors_per_class)
        masc_words, masc_scores = list(zip(*masc_words_and_scores))
        neut_words, neut_scores = list(zip(*neut_words_and_scores))
        fem_words, fem_scores = list(zip(*fem_words_and_scores))
        masc_vecs, fem_vecs = get_vectors(masc_words, self.model), get_vectors(fem_words, self.model)
        neut_vecs = get_vectors(neut_words, self.model)

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
        return X_dev, Y_dev, X_train, Y_train, X_test, Y_test, all_significantly_biased_vecs, \
               all_significantly_biased_labels, vecs

    def debias_embedding_table(self):
        """
        debiases the embedding table according to the chosen method
        :return: the debiased embedding table
        """
        # self.load_and_prepare_embedding_table()
        X_dev, Y_dev, X_train, Y_train, X_test, Y_test, all_significantly_biased_vecs, \
        all_significantly_biased_labels, vecs = self.debias_inlp_preparation()
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

        debiased_embeddings = (P.dot(vecs.T)).T
        self.save(debiased_embeddings)
        return debiased_embeddings


class DebiasBlukbasyManager(DebiasManager):
    def __init__(self, consts_config_str):
        super().__init__(consts_config_str)
        self.E = None
        self.__prepare_data_to_debias()

    def __prepare_data_to_debias(self):
        """
        given path to dictionary, the path to the embedding table saved in get_embedding_table() and the file name to save the data,
        it prepares the embedding table in the format of <word> <embedding>/n , this is the format that debias() in debiaswe, uses.
        saves the embedding with the desired format to self.EMBEDDING_DEBIASWE_FILE
        """
        with open(self.ENG_DICT_FILE, 'r') as dict_file, open(self.EMBEDDING_DEBIASWE_FILE, 'w') as dest_file:
            eng_dictionary = json.load(dict_file)
            for w, i in eng_dictionary.items():
                dest_file.write(w + " " + ' '.join(map(str, self.non_debiased_embeddings[i, :])) + "\n")

    def get_gender_direction(self):
        """
        :return: a vector represents the gender direction
        """
        with open(DEFINITIONAL_FILE, "r") as f:
            defs = json.load(f)
        return we.doPCA(defs, self.E).components_[0]

    def debias_embedding_table(self):
        """
        debiases the nematus embedding table that was created through the
        learning phase and saved in prepare_data_to_debias()
        :return: the debiased embedding table
        """
        self.E = we.WordEmbedding(self.EMBEDDING_DEBIASWE_FILE)
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
        self.save(self.E.vecs)
        return self.E.vecs

# if __name__ == '__main__':
#     print("hello world")
#     CONSTS_CONFIG_STR = "{'USE_DEBIASED': 0, 'LANGUAGE': 0, 'COLLECT_EMBEDDING_TABLE': 0, 'PRINT_LINE_NUMS': 0, 'DEBIAS_METHOD': 1}"
#     debias_manager = DebiasManager.get_manager_instance(CONSTS_CONFIG_STR)
#     debiased_embedding = debias_manager.debias_embedding_table()
#     print(np.shape(debiased_embedding))
#     print(debiased_embedding)
#     debias_manager.debias_sanity_check(debiased_embedding_table=debiased_embedding)
