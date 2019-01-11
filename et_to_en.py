import os
import string
import pickle
from collections import Counter
import argparse

import numpy as np
np.random.seed(42)
import tqdm
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu

EN_DOC_PATH = os.path.join('Final_Corpora_Et-En',
                         'europarl-v7.et-en.en')
ET_DOC_PATH = os.path.join('Final_Corpora_Et-En',
                         'europarl-v7.et-en.et')
TRAIN_TEST_RATIO = 0.9

MAX_WORD_LEN = 20

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help="train / evaluate")
    return parser.parse_args()

def load_doc(path):
    corpus = open(path, 'r', encoding='utf-8')
    return corpus.readlines()

def get_min_max_length(sentences):
    lengths = [len(s.split()) for s in sentences]
    return min(lengths), max(lengths)

def cleanup_sentences(en_sentences, et_sentences, max_word_len=MAX_WORD_LEN):
    new_en_sentences = []
    new_et_sentences = []

    # Remove some sentences (if len(sentence.split()) > max_word_len)
    for i in range(len(en_sentences)):
        en_s = en_sentences[i].strip()
        et_s = et_sentences[i].strip()

        if ';' in en_sentences[i]:
            # Split some lines using ';'
            if en_sentences[i].count(';') == et_sentences[i].count(';'):
                split_en_sentences = en_sentences[i].split(';')
                split_et_sentences = et_sentences[i].split(';')

                for i in range(len(split_en_sentences)):
                    if len(split_en_sentences[i].split()) <= max_word_len and \
                            len(split_et_sentences[i].split()) <= max_word_len:
                        new_en_sentences.append(split_en_sentences[i])
                        new_et_sentences.append(split_et_sentences[i])
                continue

        if len(en_s.split()) <= max_word_len and len(et_s.split()) <= max_word_len:
            new_en_sentences.append(en_s)
            new_et_sentences.append(et_s)

    en_sentences = new_en_sentences
    et_sentences = new_et_sentences
    new_en_sentences = []
    new_et_sentences = []
    trans_table = str.maketrans('', '', string.punctuation)
    for i in range(len(en_sentences)):
        en_s = en_sentences[i]
        et_s = et_sentences[i]

        # lower case
        en_s = en_s.lower()
        et_s = et_s.lower()
        # tokenize
        en_s = en_s.split()
        et_s = et_s.split()
        # remove punctuations
        en_s = [word.translate(trans_table) for word in en_s]
        et_s = [word.translate(trans_table) for word in et_s]

        # remove empty word
        en_s = [word for word in en_s if word != '']
        et_s = [word for word in et_s if word != '']

        # Remove Empty lines
        if len(en_s) == 0 or len(et_s) == 0:
            continue

        new_en_sentences.append(' '.join(en_s))
        new_et_sentences.append(' '.join(et_s))

    return new_en_sentences, new_et_sentences


def build_vocab_counter(sentences):
    vocab = Counter()
    for s in sentences:
        words = s.split()
        vocab.update(words)
    return vocab


def trim_vocab_counter(vocab: Counter, min_occurance):
    words = [k for k, occ, in vocab.items() if occ >= min_occurance]
    return set(words)


def build_vocab(vocab: set):
    vocab_dict = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3}
    vocab_size = len(vocab)
    for _ in range(vocab_size):
        word = vocab.pop()
        vocab_dict[word] = len(vocab_dict)
    return vocab_dict


def encode_sentences(sentences, vocab, max_len):
    encoded_sentences = np.zeros([len(sentences), max_len+1], dtype=int)
    sequence_lengths = np.zeros([len(sentences),], dtype=int)
    for i, s in enumerate(sentences):
        word_sequence = s.split()
        sequence_lengths[i] = len(word_sequence) + 1
        for j, word in enumerate(word_sequence):
            encoded_sentences[i, j] = vocab.get(word, 2)  # 2 means '<UNK>'
        encoded_sentences[i, sequence_lengths[i]-1] = 1  # 1 means '<EOS>'
    return encoded_sentences, sequence_lengths


def enc_dec_model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')

    target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
    max_target_len = tf.reduce_max(target_sequence_length)
    #max_target_len = tf.constant(MAX_WORD_LEN, dtype=tf.int32)
    return inputs, targets, target_sequence_length, max_target_len


def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    # get '<GO>' id
    go_id = target_vocab_to_int['<GO>']

    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    after_concat = tf.concat([tf.fill([batch_size, 1], go_id), after_slice], 1)
    print(after_concat)
    return after_concat


def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob,
                   source_vocab_size,
                   encoding_embedding_size):
    """
    :return: tuple (RNN output, RNN state)
    """
    embed = tf.contrib.layers.embed_sequence(rnn_inputs,
                                             vocab_size=source_vocab_size,
                                             embed_dim=encoding_embedding_size)

    stacked_cells = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size), keep_prob) for _ in range(num_layers)])

    outputs, state = tf.nn.dynamic_rnn(stacked_cells,
                                       embed,
                                       dtype=tf.float32)
    return outputs, state


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input,
                         target_sequence_length, max_summary_length,
                         output_layer, keep_prob):
    """
    Create a training process in decoding layer
    :return: BasicDecoderOutput containing training logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,
                                             output_keep_prob=keep_prob)

    # for only input layer
    helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input,
                                               target_sequence_length)

    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                              helper,
                                              encoder_state,
                                              output_layer)

    # unrolling the decoder layer
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                      impute_finished=True,
                                                      maximum_iterations=max_summary_length)
    return outputs


def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    """
    Create a inference process in decoding layer
    :return: BasicDecoderOutput containing inference logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,
                                             output_keep_prob=keep_prob)

    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                                                      tf.fill([batch_size], start_of_sequence_id),
                                                      end_of_sequence_id)

    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                              helper,
                                              encoder_state,
                                              output_layer)

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                      impute_finished=True,
                                                      maximum_iterations=max_target_sequence_length)
    return outputs


def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):
    """
    Create decoding layer
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    target_vocab_size = len(target_vocab_to_int)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(num_layers)])

    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(target_vocab_size)
        train_output = decoding_layer_train(encoder_state,
                                            cells,
                                            dec_embed_input,
                                            target_sequence_length,
                                            max_target_sequence_length,
                                            output_layer,
                                            keep_prob)

    with tf.variable_scope("decode", reuse=True):
        infer_output = decoding_layer_infer(encoder_state,
                                            cells,
                                            dec_embeddings,
                                            target_vocab_to_int['<GO>'],
                                            target_vocab_to_int['<EOS>'],
                                            max_target_sequence_length,
                                            target_vocab_size,
                                            output_layer,
                                            batch_size,
                                            keep_prob)

    return train_output, infer_output


def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence model
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    enc_outputs, enc_states = encoding_layer(input_data,
                                             rnn_size,
                                             num_layers,
                                             keep_prob,
                                             source_vocab_size,
                                             enc_embedding_size)

    dec_input = process_decoder_input(target_data,
                                      target_vocab_to_int,
                                      batch_size)

    train_output, infer_output = decoding_layer(dec_input,
                                                enc_states,
                                                target_sequence_length,
                                                max_target_sentence_length,
                                                rnn_size,
                                                num_layers,
                                                target_vocab_to_int,
                                                target_vocab_size,
                                                batch_size,
                                                keep_prob,
                                                dec_embedding_size)

    return train_output, infer_output


class ET_to_EN_Translator:

    def __init__(self):
        self._prepare_dataset()
        self.build_model()

    def _prepare_dataset(self):
        en_sentences = load_doc(EN_DOC_PATH)
        et_sentences = load_doc(ET_DOC_PATH)
        print('Raw English Data: sentences={}, min_words={}, max_words={}'.format(
            len(en_sentences), *get_min_max_length(en_sentences)
            ), flush=True)
        print('Raw Estonian Data: sentences={}, min_words={}, max_words={}'.format(
            len(et_sentences), *get_min_max_length(et_sentences)
            ), flush=True)

        # Load Clean-up sentences
        if not os.path.exists('en_sentences.pkl') or not os.path.exists('et_sentences.pkl'):
            en_sentences, et_sentences = cleanup_sentences(en_sentences, et_sentences, MAX_WORD_LEN)
            with open('en_sentences.pkl', 'wb') as fout:
                pickle.dump(en_sentences, fout)
            with open('et_sentences.pkl', 'wb') as fout:
                pickle.dump(et_sentences, fout)

        en_sentences = pickle.load(open('en_sentences.pkl', 'rb'))
        et_sentences = pickle.load(open('et_sentences.pkl', 'rb'))

        en_min_len, self.en_max_len = get_min_max_length(en_sentences)
        et_min_len, self.et_max_len = get_min_max_length(et_sentences)

        print('English Data: sentences={}, min_words={}, max_words={}'.format(
                len(en_sentences), en_min_len, self.en_max_len), flush=True)
        print('Estonian Data: sentences={}, min_words={}, max_words={}'.format(
                len(et_sentences), et_min_len, self.et_max_len), flush=True)

        test_idx_from = int(len(en_sentences) * TRAIN_TEST_RATIO)
        self.train_en_sentences = en_sentences[:test_idx_from]
        self.test_en_sentences = en_sentences[test_idx_from:]
        self.train_et_sentences = et_sentences[:test_idx_from]
        self.test_et_sentences = et_sentences[test_idx_from:]

        # Load Vocab
        if not os.path.exists('en_vocab.pkl'):
            # Build Vocab from train sentences
            en_vocab = build_vocab_counter(self.train_en_sentences)
            print('English(train data) Vocab size:', len(en_vocab), flush=True)
            en_vocab = trim_vocab_counter(en_vocab, min_occurance=10)

            with open('en_vocab.pkl', 'wb') as fout:
                pickle.dump(build_vocab(en_vocab), fout)

        if not os.path.exists('et_vocab.pkl'):
            # Build Vocab from train sentences
            et_vocab = build_vocab_counter(self.train_et_sentences)
            print('Estonian(train data) Vocab size:', len(et_vocab), flush=True)
            et_vocab = trim_vocab_counter(et_vocab, min_occurance=10)

            with open('et_vocab.pkl', 'wb') as fout:
                pickle.dump(build_vocab(et_vocab), fout)

        self.en_vocab = pickle.load(open('en_vocab.pkl', 'rb'))
        print('Trimmed English(train data) Vocab size:', len(self.en_vocab), flush=True)
        self.et_vocab = pickle.load(open('et_vocab.pkl', 'rb'))
        print('Trimmed Estonian(train data) Vocab size:', len(self.et_vocab), flush=True)

        self.reverse_en_vocab = {v: k for k, v in self.en_vocab.items()}
        self.reverse_et_vocab = {v: k for k, v in self.et_vocab.items()}

        self.train_et, self.train_et_seqlen = encode_sentences(self.train_et_sentences, self.et_vocab, self.et_max_len)
        self.train_en, self.train_en_seqlen = encode_sentences(self.train_en_sentences, self.en_vocab, self.en_max_len)

        # Shuffle train data
        p = np.random.permutation(len(self.train_et))
        self.train_et = self.train_et[p]
        self.train_et_seqlen = self.train_et_seqlen[p]
        self.train_en = self.train_en[p]
        self.train_en_seqlen = self.train_en_seqlen[p]

        self.test_et, self.test_et_seqlen = encode_sentences(self.test_et_sentences, self.et_vocab, self.et_max_len)
        self.test_en, self.test_en_seqlen = encode_sentences(self.test_en_sentences, self.en_vocab, self.en_max_len)

    def int_seqs_to_word_seqs(self, int_sequences, reverse_vocab):
        word_sequences = []
        for int_seq in int_sequences:
            word_seq = []
            for idx in int_seq:
                if idx == 0 or idx == 1:  # 0: PAD, 1: EOS
                    break
                if idx == 2:  # 2: UNK
                    continue
                word_seq.append(reverse_vocab[idx])
            word_sequences.append(word_seq)
        return word_sequences

    def build_model(self):

        self.source_vocab_to_int = self.et_vocab
        self.target_vocab_to_int = self.en_vocab

        self.epochs = 10
        self.batch_size = 128

        self.rnn_size = 512
        self.num_layers = 3

        self.encoding_embedding_size = 1024
        self.decoding_embedding_size = 1024

        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            self.input_data, self.targets, self.target_sequence_length, self.max_target_sequence_length = enc_dec_model_inputs()
            self.label_targets = tf.placeholder(tf.int32, [None, None], name='label_targets')
            self.tf_lr = tf.placeholder(tf.float32, name='lr_rate')
            self.tf_keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            train_logits, inference_logits = seq2seq_model(tf.reverse(self.input_data, [-1]),
                                                           self.targets,
                                                           self.tf_keep_prob,
                                                           self.batch_size,
                                                           self.target_sequence_length,
                                                           self.max_target_sequence_length,
                                                           len(self.source_vocab_to_int),
                                                           len(self.target_vocab_to_int),
                                                           self.encoding_embedding_size,
                                                           self.decoding_embedding_size,
                                                           self.rnn_size,
                                                           self.num_layers,
                                                           self.target_vocab_to_int)

            self.training_logits = tf.identity(train_logits.rnn_output, name='logits')
            self.inference_logits = tf.identity(inference_logits.sample_id, name='predictions')
            # https://www.tensorflow.org/api_docs/python/tf/sequence_mask
            # - Returns a mask tensor representing the first N positions of each cell.
            masks = tf.sequence_mask(self.target_sequence_length, self.max_target_sequence_length, dtype=tf.float32, name='masks')
            with tf.name_scope("optimization"):
                # Loss function - weighted softmax cross entropy
                self.cost = tf.contrib.seq2seq.sequence_loss(
                    self.training_logits,
                    self.label_targets,
                    masks)

                # Optimizer
                optimizer = tf.train.AdamOptimizer(self.tf_lr)

                # Gradient Clipping
                gradients = optimizer.compute_gradients(self.cost)
                capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if
                                    grad is not None]
                self.train_op = optimizer.apply_gradients(capped_gradients)


    def train(self):

        with tf.Session(graph=self.train_graph) as sess:
            save_path = './model'
            sess.run(tf.global_variables_initializer())
            self.train_source = self.train_et
            self.train_target = self.train_en
            self.train_target_len = self.train_en_seqlen
            self.test_source = self.test_et
            self.test_target = self.test_en
            self.test_target_len = self.test_en_seqlen

            learning_rate = 1e-3
            keep_probability = 0.8
            display_step = 500

            mini_test_source = self.test_source[:self.batch_size]
            mini_test_target = self.test_target[:self.batch_size]
            mini_test_target_len = self.test_target_len[:self.batch_size]
            total_train_steps = len(self.train_source) // self.batch_size  # discard residual batch

            for epoch_i in range(self.epochs):

                # Train
                pbar = tqdm.tqdm(range(total_train_steps))
                #pbar = tqdm.tqdm(range(300))
                for i in pbar:
                    source_batch = self.train_source[i*self.batch_size:(i+1)*self.batch_size]
                    target_batch = self.train_target[i*self.batch_size:(i+1)*self.batch_size]
                    target_len_batch = self.train_target_len[i*self.batch_size: (i+1)*self.batch_size]
                    _, loss = sess.run([self.train_op, self.cost],
                                        {self.input_data: source_batch,
                                         self.targets: target_batch,
                                         self.label_targets: target_batch[:,:max(target_len_batch)],
                                         self.tf_lr: learning_rate,
                                         self.target_sequence_length: target_len_batch,
                                         self.tf_keep_prob: keep_probability})
                    pbar.set_description('Epoch {:>3} tLoss: {:>6.4f}'.format(epoch_i, loss))

                    if i % display_step == 0:
                        batch_train_logits = sess.run(
                            self.inference_logits,
                            {self.input_data: source_batch,
                             self.target_sequence_length: target_len_batch,
                             self.tf_keep_prob: 1.0})

                        batch_test_logits = sess.run(
                            self.inference_logits,
                            {self.input_data: mini_test_source,
                             self.target_sequence_length: mini_test_target_len,
                             self.tf_keep_prob: 1.0})
                        print('\n Train Samples >>>')
                        for k in range(1, 4):
                            print('[Source_%d]'%k, ' '.join(self.int_seqs_to_word_seqs([source_batch[k]], self.reverse_et_vocab)[0]))
                            print('[Target_%d]'%k, ' '.join(self.int_seqs_to_word_seqs([target_batch[k]], self.reverse_en_vocab)[0]))
                            print('[Predict_%d]'%k, ' '.join(self.int_seqs_to_word_seqs([batch_train_logits[k]], self.reverse_en_vocab)[0]))
                            print()
                        print('\n Test Samples >>>') 
                        for k in range(1, 4):
                            print('[Source_%d]'%k, ' '.join(self.int_seqs_to_word_seqs([mini_test_source[k]], self.reverse_et_vocab)[0]))
                            print('[Target_%d]'%k, ' '.join(self.int_seqs_to_word_seqs([mini_test_target[k]], self.reverse_en_vocab)[0]))
                            print('[Predict_%d]'%k, ' '.join(self.int_seqs_to_word_seqs([batch_test_logits[k]], self.reverse_en_vocab)[0]))
                            print()
                        train_pred_seqs = self.int_seqs_to_word_seqs(batch_train_logits, self.reverse_en_vocab)
                        train_real_seqs = self.int_seqs_to_word_seqs(target_batch, self.reverse_en_vocab)
                        train_real_seqs = [[seq] for seq in train_real_seqs]
                        train_bleu = corpus_bleu(train_real_seqs, train_pred_seqs, weights=(0.25, 0.25, 0.25, 0.25))

                        test_pred_seqs = self.int_seqs_to_word_seqs(batch_test_logits, self.reverse_en_vocab)
                        test_real_seqs = self.int_seqs_to_word_seqs(mini_test_target , self.reverse_en_vocab)
                        test_real_seqs = [[seq] for seq in test_real_seqs]
                        test_bleu = corpus_bleu(test_real_seqs, test_pred_seqs, weights=(0.25, 0.25, 0.25, 0.25))

                        print('\n[Epoch {:>3}] Train BLEU: {:>6.4f}, mini Test BLEU: {:>6.4f}, tLoss: {:>6.4f}'
                                .format(epoch_i, train_bleu, test_bleu, loss), flush=True)

                # Test
                test_logits = []
                total_test_steps = len(self.test_source) // self.batch_size
                pbar = tqdm.tqdm(range(total_test_steps))
                pbar.set_description('Test...')
                for i in pbar:
                    source_batch = self.test_source[i*self.batch_size:(i+1)*self.batch_size]
                    target_batch = self.test_target[i*self.batch_size:(i+1)*self.batch_size]
                    target_len_batch = self.test_target_len[i*self.batch_size:(i+1)*self.batch_size]

                    batch_test_logits = sess.run(
                        self.inference_logits,
                        {self.input_data: source_batch,
                         self.target_sequence_length: target_len_batch,
                         self.tf_keep_prob: 1.0})

                    test_logits.extend(batch_test_logits)

                pred_seqs = self.int_seqs_to_word_seqs(test_logits, self.reverse_en_vocab)
                real_seqs = [[s.split()] for s in self.test_en_sentences][:len(pred_seqs)]
                print('\n[Test] BLEU-1gram: %f' % corpus_bleu(real_seqs, pred_seqs, weights=(1.0, 0, 0, 0)))
                print('\n[Test] BLEU-2gram: %f' % corpus_bleu(real_seqs, pred_seqs, weights=(0.5, 0.5, 0, 0)))
                print('\n[Test] BLEU-3gram: %f' % corpus_bleu(real_seqs, pred_seqs, weights=(0.333, 0.333, 0.333, 0)))
                print('\n[Test] BLEU-4gram: %f' % corpus_bleu(real_seqs, pred_seqs, weights=(0.25, 0.25, 0.25, 0.25)))

            # Save Model
            saver = tf.train.Saver()
            os.makedirs(save_path, exist_ok=True)
            saver.save(sess, save_path)
            print('Model Trained and Saved')


if __name__ == '__main__':
    args = argument_parser()
    translator = ET_to_EN_Translator()
    if args.mode == 'train':
        translator.train()
    else:
        print("argument error")
