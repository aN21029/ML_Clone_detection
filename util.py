import json
import solcx
import solcast
import os
from random import shuffle
from gensim.models import Word2Vec
import itertools
from attention import apply_attention
import numpy as np
import tensorflow as tf
from random import randint
from siamese import train_model


TRAIN_SPLIT, TEST_SPLIT = 0.7, 0.3
NUM_DATA_REPEAT = 10
DEFAULT_SIMILARITY_THRESHOLD = 0.8
SOLIDITY_COMPILER_VERSIONS = [
    "0.4.1","0.4.2","0.4.3","0.4.4","0.4.5","0.4.6","0.4.7","0.4.8","0.4.9","0.4.10","0.4.11","0.4.12","0.4.13","0.4.14","0.4.15","0.4.16","0.4.17","0.4.18","0.4.19","0.4.20","0.4.21","0.4.22","0.4.23","0.4.24","0.4.25","0.4.26",
    "0.5.0","0.5.1","0.5.2","0.5.3","0.5.4","0.5.5","0.5.6","0.5.7","0.5.8","0.5.9","0.5.10","0.5.11","0.5.12","0.5.13","0.5.14","0.5.15","0.5.16","0.5.17",
    "0.6.0","0.6.1","0.6.2","0.6.3","0.6.4","0.6.5","0.6.6","0.6.7","0.6.8","0.6.9","0.6.10","0.6.11","0.6.12",
    "0.7.0","0.7.1","0.7.2","0.7.3","0.7.4","0.7.5", "0.7.6",
    "0.8.0","0.8.1","0.8.2","0.8.3","0.8.4","0.8.5","0.8.6","0.8.7","0.8.8","0.8.9","0.8.10","0.8.11","0.8.12","0.8.13","0.8.14","0.8.15","0.8.16","0.8.17","0.8.18","0.8.19","0.8.20","0.8.21","0.8.22"
]


class CodeLevelAnalyzer:
    """
    Represents a contract-level clone detection analyzer

    Attrbibutes:
        data: the dataset being analyzed
        labels: all the labels corresponding to each piece of data 
    """
    def __init__(self, data):
        # for version in SOLIDITY_COMPILER_VERSIONS:
        #     os.system("solc-select install {}".format(version))

        self.data = data
        self.contracts_analyzed, self.invalid_contracts = 0, 0
        self.embedding_matrix, self.embedding_matrix_labels = self._embed_code()

        # 90-10 train test split
        split = round(TRAIN_SPLIT*len(self.embedding_matrix))
        # need to make sure the training set has an even number of samples, which is necessary because of the way the ancho-positive pairs inside the siamese network are formed
        split += split % 2
        shuffle(self.embedding_matrix)
        self.train_embeddings, self.test_embeddings = self.embedding_matrix[:split], self.embedding_matrix[split:]
        self.train_labels, self.test_labels = self.embedding_matrix_labels[:split], self.embedding_matrix_labels[split:]

        self.similarity_threshold = DEFAULT_SIMILARITY_THRESHOLD # can be changed depending on what threshold value gets passed into _train_model()
        self.embedding_model, self.siamese_network = self._train_model()
    
    def _embed_code(self):
        embeddings = []
        labels = []

        # for i in range(len(self.data)):
        for i in range(5): # ONLY LOOPING 5 TIMES FOR TESTING, ABOVE FOR LOOP IS THE ACTUAL LOOP
            ast = AST(self.data[i]["source_code"]) # some ASTs may be invalid if compilation results in error
            if ast.root:
                embedding = CodeEmbedding(ast)
                embeddings.append(embedding.vector)
                # cur_labels = [1 if label in self.data[i]["slither"] else 0 for label in range(6)]
                cur_labels = self.data[i]["slither"]
                labels.append(cur_labels)
                self.contracts_analyzed += 1
            else:
                self.invalid_contracts += 1
        equalize_vector_lengths(embeddings) # some paths may be longer than others, make sure vector lengths are equal to be able to plug into siamese network
        return np.array(embeddings), labels
        # return np.concatenate((embeddings, labels), axis=1) # important to convert to np array AFTER embedding, because code embeddings may be different length, so need to add padding to smaller vectors first 
        
    def _train_model(self):
        embedding_model, siamese_model = train_model(np.repeat(self.train_embeddings, NUM_DATA_REPEAT, axis=0)) # repeat each piece of data a number of times for better mining of positive-anchor pairs
        return embedding_model, siamese_model
    
    def _cosine_similarity(self, embedding1, embedding2):
        return tf.keras.losses.CosineSimilarity(axis=1)(embedding1, embedding2).numpy()
    
    def _aggregate_test_data_by_class(self):
        data_groups = {i:[] for i in range(6)}
        for i in range(len(self.test_embeddings)):
            for class_label in self.test_labels[i]:
                data_groups[class_label].append(self.test_embeddings[i])
        return data_groups

    def _predict_test_set_classes(self, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD, similarity_metric="cosine"):
        def cosine_similarity_matrix(embeddings):
            return np.array([np.dot(embeddings,embedding)/(np.linalg.norm(embeddings,axis=1)*np.linalg.norm(embedding)) for embedding in embeddings])
        
        self.similarity_threshold = similarity_threshold

        model_embeddings = self.embedding_model.predict(self.train_embeddings)
        similarities = cosine_similarity_matrix(model_embeddings)
        similarity_mask = similarities > similarity_threshold
        np.fill_diagonal(similarity_mask, False) # make sure not to include the similarity of piece of data with itself

        predictions = []
        # print(similarity_mask)
        for i in  range(len(model_embeddings)):
            is_similar = similarity_mask[i]
            predicted_labels = set()

            for j in range(len(is_similar)):
                if is_similar[j]:
                    for label in self.train_labels[j]:
                        predicted_labels.add(label)
            predictions.append(list(predicted_labels))

        return predictions

        # return predictions

        # embedding(tf.reshape(anchors[i], (-1, len(anchors[0]))))
        # class_predictions = []
        # embedding_length = len(self.test_embeddings[0])
        # test_data = self._aggregate_test_data_by_class()
        # random = lambda upper_bound: randint(0, upper_bound - 1)
        # len_class = lambda class_label: len(test_data[class_label])
        # format_test = lambda i: tf.data.Dataset.from_tensor_slices([[self.test_embeddings[i,:embedding_length // 3]],
        #                                                             [self.test_embeddings[i,embedding_length // 3: 2 * (embedding_length // 3)]],
        #                                                             [self.test_embeddings[i,2 * (embedding_length // 3):]]])
        # format_class = lambda class_label: tf.data.Dataset.from_tensor_slices([[test_data[class_label][random(len_class(class_label))][:embedding_length // 3]],
        #                                                                        [test_data[class_label][random(len_class(class_label))][embedding_length // 3: 2 * (embedding_length // 3)]],
        #                                                                        [test_data[class_label][random(len_class(class_label))][2 * (embedding_length // 3):]]])
        # for i in range(len(self.test_embeddings)):
        #     # make sure input to the model is formatted correctly
        #     model_input = format_test(i)
        #     cur_prediction = self.siamese_network.model.predict(model_input)

        #     cur_class_predictions = []
        #     for class_label in test_data:
        #         if test_data[class_label]:
        #             sample = format_class(class_label)
        #             class_prediction = self.siamese_network.model.predict(sample)

        #             if abs(self._cosine_similarity(cur_prediction, class_prediction)) > similarity_threshold:
        #                 cur_class_predictions.append(class_label)
        #     class_predictions.append(cur_class_predictions)
        
        # return class_predictions

    # evaluation metrics to measure accuracy of model

    def _get_positives(self, test_predictions):
        true_positives = {i: 0 for i in range(6)}
        false_positives = {i: 0 for i in range(6)}

        for i in range(len(test_predictions)):
            for label in test_predictions[i]:
                if label in self.test_labels[i]:
                    true_positives[label] += 1
                else:
                    false_positives[label] += 1
        return true_positives, false_positives
                    
    
    def _get_negatives(self, test_predictions):
        classes = [0,1,2,3,4,5]
        true_negatives = {i: 0 for i in range(6)}
        false_negatives = {i: 0 for i in range(6)}

        for i in range(len(test_predictions)):
            pred_negatives = [class_label for class_label in classes if class_label not in test_predictions[i]]
            for label in pred_negatives:
                if label not in self.test_labels[i]:
                    true_negatives[label] += 1
                else:
                    false_negatives[label] += 1
        return true_negatives, false_negatives

    def _get_recall_scores(self, true_positives, false_negatives):
        recall_scores = {i: 0 for i in range(6)}

        for label in recall_scores:
            TP = true_positives[label]
            FN = false_negatives[label]
            recall_scores[label] = TP / (TP + FN) if TP + FN > 0 else float("nan") # make sure there is no division by 0
        return recall_scores

    def _get_precision_scores(self, true_positives, false_positives):
        recall_scores = {i: 0 for i in range(6)}

        for label in recall_scores:
            TP = true_positives[label]
            FP = false_positives[label]
            recall_scores[label] = TP / (TP + FP) if TP + FP > 0 else float("nan")
        return recall_scores

    def _get_f1_scores(self, recall_scores, precision_scores):
        f1_scores = {i: 0 for i in range(6)}

        for label in f1_scores:
            recall = recall_scores[label]
            precision = precision_scores[label]

            f1_scores[label] = 2*precision*recall / (precision + recall)
        return f1_scores

    def model_statistics(self):
        # embedding.summary()

        # for layer in embedding.layers:
        #     print(layer.input_shape)
        
        # siamese_network.summary()

        # for layer in siamese_network.layers:
        #     print(layer.input_shape)
        predictions = self._predict_test_set_classes(similarity_threshold=self.similarity_threshold)
        true_positives, false_positives = self._get_positives(predictions)
        true_negatives, false_negatives = self._get_negatives(predictions)
        precision_scores, recall_scores = self._get_precision_scores(true_positives, false_positives), self._get_recall_scores(true_positives, false_negatives)
        f1_scores = self._get_f1_scores(recall_scores, precision_scores)

        output = """
        contracts_analyzed: {}
        invalid_contracts: {}
        true_positives: {}
        true_negatives: {}
        false_positive: {}
        false_negative: {}
        precision: {}
        recall: {}
        f1_score: {}""".replace(" ", "").format(self.contracts_analyzed,self.invalid_contracts,true_positives,true_negatives,false_positives,false_negatives,precision_scores, recall_scores, f1_scores)

        return output

class AST:
    """
    An abstract syntax tree representation of a solidity smart contract

    Attributes:
        root: the root of the AST representation
    """
    def __init__(self, contract):
        # Each field in a node follows the expected AST grammar: https://docs.soliditylang.org/en/latest/grammar.html
        # more info: https://pypi.org/project/py-solc-ast/
        self.root = self._parse_AST(contract) 



    def _parse_AST(self, contract):
        """
        Given a string representation of a smart contract, parses it into an AST
        """
        input_filename = "setting.json"
        contract_filename = "contract.sol"

        # first, fetch which solidity version the code should be compiled in
        try:
            solidity_version = contract.split("\n")[0].split(" ")[2].lstrip("^").rstrip(";")
        except IndexError:
            return None

        # and make sure it is installed and is the version being used to compile current contract
        # os.system("solc-select install {}".format(solidity_version))
        os.system("solc-select use {}".format(solidity_version))

        # USED FOR TESTING PURPOSES (used in lines below to write json details to files)
        input_json_filename = "input.json"
        output_json_filename = "output.json"

        # write the contract data to a file so it can be referred as a source when compiling AST
        with open(contract_filename, "w", encoding='utf-8') as contract_file:
            contract_file.write(contract)

        # compile contract to standard JSON input format
        with open(input_filename, "w", encoding='utf-8') as inputFile:
            inputFile.write(JSON_STANDARD_INPUT(contract_filename, contract))
        input_json = json.load(open(input_filename))

        # write the contents to a file if you would like to see what format looks like
        with open(input_json_filename, "w") as file:
            file.write(json.dumps(input_json))

        # Output json file
        try:
            output_json = solcx.compile_standard(input_json)

            with open(output_json_filename, "w") as astFile:
                astFile.write(json.dumps(output_json))
            
            # extract AST from output json file format
            root = solcast.from_standard_output(output_json)[0]

            return root
        except solcx.exceptions.SolcError: # if there is an error in compilation, the tree is invalid
            return None

    def get_terminal_nodes(self):
        """
        Get all the terminal nodes in the AST
        """
        children = np.array(self.root.children(), dtype=object)

        is_terminal = np.vectorize(lambda node: node.children(depth=1) == [])

        terminal_node_indices = is_terminal(children)
        return children[terminal_node_indices]
    

    # no need to use numpy here, or for any functions that calculate a path along a tree.
    # Calculating paths is at most O(H), where H is the height of the tree. Which, even for trees with >1000 nodes, tends to be around 8-15
    # Creating numpy arrays has a large overhead so it may be slower for trivial sizes
    def get_path_from_root(self, node):
        root_reached = False
        path = []
        cur = node

        # start from given node, and keep traversing up the parent until the root is reached
        while not root_reached:
            path.append(cur)
            try:
                cur = cur.parent()
            except IndexError:
                root_reached = True
        
        return path[::-1]

    def find_lowest_common_ancestor(self, node1, node2):
        # Get path from both nodes to the root
        path1 = self.get_path_from_root(node1)
        path2 = self.get_path_from_root(node2)

        lowest_common = None

        # starting from the root, traverse down the tree one level at a time
        # if both paths contain the same node at the same depth, it is a common ancestor
        # the last common ancestor encoutnered in both paths is the lowest common ancestor
        for i in range(min(len(path1), len(path2))):
            if path1[i].src == path2[i].src:
                lowest_common = path1[i] 
        
        return lowest_common
    
    def _path_child_to_parent(self, child_node, parent_node):
        """
        Finds the path from a node to its parent

        precondition: parent_node is a parent of child_node
        """
        path = []
        cur = child_node

        while cur.src != parent_node.src:
            path.append(cur)
            cur = cur.parent()
        
        return path
    

    def get_path(self, node1, node2):
        """
        Finds the path from node1 to node2
        """
        lowest_common_ancestor = self.find_lowest_common_ancestor(node1, node2)

        path = self._path_child_to_parent(node1, lowest_common_ancestor)
        path.append(lowest_common_ancestor)
        path.extend(self._path_child_to_parent(node2, lowest_common_ancestor)[::-1])
        
        return path



class CodeEmbedding:
    """
    Represents a code embedding of a given AST
    """
    def __init__(self, AST, vector_size=3, window_size=4, min_count=1, workers=1):
        tokens = self._tokenize_paths(self._sample_paths(AST))
        self.model = self._build_word2vec_model(tokens, vector_size, window_size, min_count, workers) # model that trains the tokenized path vectors
        self.vector = self._get_vector(tokens)

    def _get_vector(self, paths):
        """
        Get the vector representation of an AST based on the given sampled paths
        """
        path_embeddings = self._get_path_embeddings(paths)
        equalize_vector_lengths(path_embeddings)
        initial_code_embedding = self._compress_path_matrix(path_embeddings)
        code_embedding = self._train_attention(path_embeddings, initial_code_embedding)
        return code_embedding

    # N = number of paths to sample
    def _sample_paths(self, AST, N=15000):
        terminals = AST.get_terminal_nodes()
        samples = np.random.choice(len(terminals), N*2)

        return [AST.get_path(terminals[samples[i]], terminals[samples[i + 1]]) for i in range(0, len(samples), 2)]


    
    def _tokenize_paths(self, paths):
        # take the matrix of paths, and tokenize the node type for each node in a path.
        return [[node.nodeType for node in path] for path in paths]


    def _build_word2vec_model(self, paths, vector_size=3, window_size=4, min_count=1, workers=1):
        model =  Word2Vec(vector_size=vector_size, window=window_size, min_count=min_count, workers=workers)
        model.build_vocab(paths) # prepare model vocab
        model.train(paths, total_examples=model.corpus_count, epochs=model.epochs) # train model weights
        return model


    def _get_path_embeddings(self, paths):
        # get the matrix representation of each path
        # here, the matrix representation is simply the vector embedding of each path
        
        # path_matrix = []
        # for path in paths:
        #     embeddings = [self.model.wv[node_token] for node_token in path] # get the word embedding of each node in the path, producing a 2D matrix
        #     path_vector = list(itertools.chain(*embeddings))
        #     path_matrix.append(path_vector)
        
        # return path_matrix

        # performs same function in above comment, but list comprehenshions are more efficient
        return [list(itertools.chain(*[self.model.wv[node_token] for node_token in path])) for path in paths]
    

    def _train_attention(self, paths, code_embedding):
        """
        Trains the path matrix such that the most relevant paths are given more weight
        """
        return apply_attention(paths, code_embedding)

    def _compress_path_matrix(self, paths):
        return np.sum(paths, axis=0)


class SiameseNetwork:
    def __init__(self, num_features):
        self.model = tf.keras.Sequential([
        tf.keras.layers.Dense(512),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation="softmax"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(num_features)])
    
    def _triplet_loss(self, y_pred, margin=1):
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        basic_loss = pos_dist - neg_dist + margin
        loss = tf.maximum(basic_loss, 0.0)
        return tf.reduce_mean(loss)
    
    def compile_model(self):
        self.model.compile(tf.keras.optimizers.Adam(0.0001), loss=self._triplet_loss)#, weighted_metrics=[self._triplet_loss])


def JSON_STANDARD_INPUT(contract_name, contract_contents):
    """
    Creates a json file with the standard input format (https://docs.soliditylang.org/en/latest/using-the-compiler.html#input-description)
    for a given solidity contract.

    Arguments:
        contract_name: the path to the contract file
        contract_contents: the contents of the smart contract in a string format
    
    Returns:
        A JSON string 
    """
    JSON_input = {
        "language": "Solidity",
        "sources":
        {
            contract_name:
            {
                "content": contract_contents
            }
        },
        "settings":
        {
            "outputSelection":
            {
                "*":
                {
                    "": ["ast"]
                }
            }
        }
    }

    JSON_input = json.dumps(JSON_input, indent=4)

    return JSON_input


def equalize_vector_lengths(paths):
        lengths = [len(path) for path in paths] # get lengths of each vector in the paths
        max_len = max(lengths) # get the length of the longest vector

        # add padding to shorter vectors so they have the same length. This so that the path matrix can be later compressed to form the code vector
        for i in range(len(paths)):
            paths[i] = np.pad(paths[i], (0, max_len - len(paths[i])))

