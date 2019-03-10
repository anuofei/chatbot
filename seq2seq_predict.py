import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import pickle

'-----------------------------------------------------------------------------------'
'加载字典数据，数据已经写成object'
dbfile = open('word_to_id_obj', 'rb')
source_vocab_to_int = pickle.load(dbfile)
dbfile2 = open('word_to_id_obj', 'rb')
target_vocab_to_int = pickle.load(dbfile2)
'-----------------------------------------------------------------------------------'

'加载文本向量。已经写成object'
'---------------------------------------------------------------------------------------------'
dbfile = open('all_ask_vec_shuffle_test_obj', 'rb')
source_text_to_int = pickle.load(dbfile)
dbfile2 = open('all_response_vec_shuffle_test_obj', 'rb')
target_text_to_int = pickle.load(dbfile2)
'----------------------------------------------------------------------------------------------'
dbfile3 = open('id_to_word_obj', 'rb')
source_int_to_vocab = pickle.load(dbfile3)
dbfile4 = open('id_to_word_obj', 'rb')
target_int_to_vocab = pickle.load(dbfile4)
'---------------------------------------------------------------------------------------------'
batch_size = 128

def sentence_to_seq(sentence, source_vocab_to_int):
    """
    将句子转化为数字编码
    """
    unk_id = source_vocab_to_int['UNK']
    word_id = [source_vocab_to_int.get(word, unk_id) for word in sentence]

    return word_id
ask_sentence_text = input("请输入句子：")
ask_sentence = sentence_to_seq(ask_sentence_text, source_vocab_to_int)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph('checkpoints/dev.meta')
    loader.restore(sess, tf.train.latest_checkpoint('./checkpoints'))

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_len:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_len:0')

    ask_logits = sess.run(logits, {input_data: [ask_sentence]*batch_size,
                                         target_sequence_length: [len(ask_sentence)*2]*batch_size,
                                         source_sequence_length: [len(ask_sentence)]*batch_size})[0]

print('Input::')
print('ask_words_Ids:      {}'.format([i for i in ask_sentence]))
print('ask_Words: {}'.format([source_int_to_vocab[i] for i in ask_sentence]))

print('\nPrediction')
print('response_Word_Ids:      {}'.format([i for i in ask_logits]))
print('response_Words: {}'.format([target_int_to_vocab[i] for i in ask_logits]))

print("\nFull_Sentence")
print(" ".join([target_int_to_vocab[i] for i in ask_logits]))
# 【Input】
#   ask Ids:      [2459, 134, 66, 316, 217, 41, 9, 89, 570, 87, 25, 36, 2518]
#   ask Words: ['邱', '先', '生', '刚', '才', '说', '了', '啊', '站', '起', '来', '就', '驳']
#
# 【Prediction】
#   Word Ids:      [104, 104, 7, 3]
#   response Words: ['谢', '谢', '你', 'EOS']
#
# 【Full Sentence】
# 谢 谢 你 EOS
#
# Process finished with exit code 0
# 【Input】
#   ask Ids:      [14, 10, 26, 100, 319, 41, 977, 13, 8, 1242, 311, 3555, 528, 6]
#   ask Words: ['这', '不', '人', '家', '更', '说', '咱', '们', '是', '朝', '三', '暮', '四', '的']
#
# 【Prediction】
#   Word Ids:      [5, 13, 8, 10, 8, 18, 14, 54, 3]
#   response Words: ['我', '们', '是', '不', '是', '在', '这', '里', 'EOS']
#
# 【Full Sentence】
# 我 们 是 不 是 在 这 里 EOS
# Input::
# ask_words_Ids:      [257, 25, 36, 3131, 10, 87, 977, 13, 2931, 1087, 130, 89]
# ask_Words: ['本', '来', '就', '瞅', '不', '起', '咱', '们', '梨', '园', '行', '啊']
#
# Prediction
# response_Word_Ids:      [7, 6, 69, 3]
# response_Words: ['你', '的', '!', 'EOS']
#
# Full_Sentence
# 你 的 ! EOS
#
# Process finished with exit code 0