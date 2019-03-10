import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import  pickle

'-----------------------------------------------------------------------------------'
'加载字典数据，数据已经写成object'
dbfile = open('word_to_id_obj', 'rb')
source_vocab_to_int = pickle.load(dbfile)
dbfile2 = open('word_to_id_obj', 'rb')
target_vocab_to_int = pickle.load(dbfile2)
'-----------------------------------------------------------------------------------'

'加载文本向量。已经写成object'
'---------------------------------------------------------------------------------------------'
dbfile = open('all_ask_vec_shuffle_train_obj', 'rb')
source_text_to_int = pickle.load(dbfile)
dbfile2 = open('all_response_vec_shuffle_train_obj', 'rb')
target_text_to_int = pickle.load(dbfile2)
'----------------------------------------------------------------------------------------------'

def model_inputs():

    inputs = tf.placeholder(tf.int32, [None, None], name="inputs")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    source_sequence_len = tf.placeholder(tf.int32, (None,), name="source_sequence_len")
    target_sequence_len = tf.placeholder(tf.int32, (None,), name="target_sequence_len")
    max_target_sequence_len = tf.placeholder(tf.int32, (None,), name="max_target_sequence_len")

    return inputs, targets, learning_rate, source_sequence_len, target_sequence_len, max_target_sequence_len


def encoder_layer(rnn_inputs, rnn_size, rnn_num_layers,
                  source_sequence_len, source_vocab_size, encoder_embedding_size=100):
    # 对输入的单词进行词向量嵌入
    encoder_embed = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoder_embedding_size)

    # LSTM单元
    def get_lstm(rnn_size):
        lstm = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
        return lstm

    # 堆叠rnn_num_layers层LSTM
    lstms = tf.contrib.rnn.MultiRNNCell([get_lstm(rnn_size) for _ in range(rnn_num_layers)])
    encoder_outputs, encoder_states = tf.nn.dynamic_rnn(lstms, encoder_embed, source_sequence_len,
                                                        dtype=tf.float32)
    return encoder_outputs, encoder_states


def decoder_layer_inputs(target_data, target_vocab_to_int, batch_size):

    # 去掉batch中每个序列句子的最后一个单词
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    # print('ending==\n',ending)
    # time.sleep(1)
    # 在batch中每个序列句子的前面添加”<GO>"
    decoder_inputs = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int["GO"]),
                                ending], 1)

    return decoder_inputs


def decoder_layer_train(encoder_states, decoder_cell, decoder_embed,
                        target_sequence_len, max_target_sequence_len, output_layer):


    # 生成helper对象
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed,
                                                        sequence_length=target_sequence_len,
                                                        time_major=False)
    training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                       training_helper,
                                                       encoder_states,
                                                       output_layer)

    training_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                       impute_finished=True,
                                                                       maximum_iterations=max_target_sequence_len)

    return training_decoder_outputs


def decoder_layer_infer(encoder_states, decoder_cell, decoder_embed, start_id, end_id,
                        max_target_sequence_len, output_layer, batch_size):

    start_tokens = tf.tile(tf.constant([start_id], dtype=tf.int32), [batch_size], name="start_tokens")

    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embed,
                                                                start_tokens,
                                                                end_id)
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                        inference_helper,
                                                        encoder_states,
                                                        output_layer)

    inference_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                        impute_finished=True,
                                                                        maximum_iterations=max_target_sequence_len)

    return inference_decoder_outputs


def decoder_layer(encoder_states, decoder_inputs, target_sequence_len,
                  max_target_sequence_len, rnn_size, rnn_num_layers,
                  target_vocab_to_int, target_vocab_size, decoder_embedding_size, batch_size):

    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoder_embedding_size]))
    decoder_embed = tf.nn.embedding_lookup(decoder_embeddings, decoder_inputs)

    def get_lstm(rnn_size):
        lstm = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=456))
        return lstm

    decoder_cell = tf.contrib.rnn.MultiRNNCell([get_lstm(rnn_size) for _ in range(rnn_num_layers)])

    # output_layer logits
    output_layer = tf.layers.Dense(target_vocab_size)

    with tf.variable_scope("decoder"):
        training_logits = decoder_layer_train(encoder_states,
                                              decoder_cell,
                                              decoder_embed,
                                              target_sequence_len,
                                              max_target_sequence_len,
                                              output_layer)

    with tf.variable_scope("decoder", reuse=True):
        inference_logits = decoder_layer_infer(encoder_states,
                                               decoder_cell,
                                               decoder_embeddings,
                                               target_vocab_to_int["GO"],
                                               target_vocab_to_int["EOS"],
                                               max_target_sequence_len,
                                               output_layer,
                                               batch_size)

    return training_logits, inference_logits


def seq2seq_model(input_data, target_data, batch_size,
                  source_sequence_len, target_sequence_len, max_target_sentence_len,
                  source_vocab_size, target_vocab_size,
                  encoder_embedding_size, decoder_embeding_size,
                  rnn_size, rnn_num_layers, target_vocab_to_int):

    _, encoder_states = encoder_layer(input_data, rnn_size, rnn_num_layers, source_sequence_len,
                                      source_vocab_size, encoder_embedding_size)
    decoder_inputs = decoder_layer_inputs(target_data, target_vocab_to_int, batch_size)

    training_decoder_outputs, inference_decoder_outputs = decoder_layer(encoder_states,
                                                                        decoder_inputs,
                                                                        target_sequence_len,
                                                                        max_target_sentence_len,
                                                                        rnn_size,
                                                                        rnn_num_layers,
                                                                        target_vocab_to_int,
                                                                        target_vocab_size,
                                                                        decoder_embeding_size,
                                                                        batch_size)
    return training_decoder_outputs, inference_decoder_outputs
class ConfigSet():
    def __init__(self):
        self.epochs = 10
        self.batch_size = 128
        self.rnn_size = 128
        self.rnn_num_layers = 1
        self.encoder_embedding_size = 100
        self.decoder_embedding_size = 100
        self.lr = 0.001
        self.display_step = 50
configset=ConfigSet()



def get_batches(sources, targets, batch_size):

    for batch_i in range(0, len(sources) // batch_size):

        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        targets_lengths = []
        for target in targets_batch:

            targets_lengths.append(len(target))
        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))
        yield sources_batch, targets_batch, source_lengths, targets_lengths
def seq2seq_run():
    train_graph = tf.Graph()
    with train_graph.as_default():
        inputs, targets, learning_rate, source_sequence_len, target_sequence_len, _ = model_inputs()

        max_target_sequence_len = 118
        train_logits, inference_logits = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       configset.batch_size,
                                                       source_sequence_len,
                                                       target_sequence_len,
                                                       max_target_sequence_len,
                                                       len(source_vocab_to_int),
                                                       len(target_vocab_to_int),
                                                       configset.encoder_embedding_size,
                                                       configset.decoder_embedding_size,
                                                       configset.rnn_size,
                                                       configset.rnn_num_layers,
                                                       target_vocab_to_int)

        training_logits = tf.identity(train_logits.rnn_output, name="logits")
        inference_logits = tf.identity(inference_logits.sample_id, name="predictions")

        masks = tf.sequence_mask(target_sequence_len, max_target_sequence_len, dtype=tf.float32, name="masks")

        with tf.name_scope("optimization"):
            cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)

            optimizer = tf.train.AdamOptimizer(learning_rate)
            gradients = optimizer.compute_gradients(cost)
            clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(clipped_gradients)
    '-----------------------------------------------------------------------------------------'
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(configset.epochs):
            for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                    get_batches(source_text_to_int, target_text_to_int, configset.batch_size)):

                _, loss = sess.run(
                    [train_op, cost],
                    {inputs: source_batch,
                     targets: target_batch,
                     learning_rate: configset.lr,
                     source_sequence_len: sources_lengths,
                     target_sequence_len: targets_lengths})

                if batch_i % configset.display_step == 0 and batch_i > 0:
                    batch_train_logits = sess.run(
                        inference_logits,
                        {inputs: source_batch,
                         source_sequence_len: sources_lengths,
                         target_sequence_len: targets_lengths})
                    print('Epoch {:>3} Batch {:>4}/{} - Loss: {:>6.4f}'
                          .format(epoch_i, batch_i, len(source_text_to_int) // configset.batch_size, loss))

                    # Save Model
        saver = tf.train.Saver()
        saver.save(sess, "checkpoints/dev")
        print('Model Trained and Saved')
if __name__=='__main__':
    seq2seq_run()







