from collections import Counter
import pickle
import numpy as np
filename='all_convs.txt'
def read_file(filename):
    """读取文件数据"""
    contents = []
    with open(filename,encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            contents.append(line)

    dbfile = open('all_convs_obj', 'wb')
    pickle.dump(contents, dbfile)
    dbfile.close()
    return contents

def build_vocab(vocab_size=5000):

    START_VOCABULART=['PAD','UNK','GO','EOS']
    """根据训练集构建词汇表，存储"""
    dbfile = open('all_convs_obj', 'rb')
    all_convs_list = pickle.load(dbfile)
    all_data = []
    for content in all_convs_list:
        for content2 in content:
            for content3 in content2.replace(' ',''):
                all_data.extend(content3)
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    words = START_VOCABULART + list(words)
    print('words==', words)
    open('vocab.txt', mode='w',encoding='utf-8').write('\n'.join(words) + '\n')
    dbfile = open('vocab_obj', 'wb')
    pickle.dump(words, dbfile)

def read_vocab():
    with open('vocab.txt', encoding='utf8') as file:
        vocabulary_list = [k.strip() for k in file.readlines()]
    word2id_dict = dict([(b, a) for a, b in enumerate(vocabulary_list)])
    print('word2id_dict===', word2id_dict)
    dbfile = open('word_to_id_obj', 'wb')
    pickle.dump(word2id_dict, dbfile)
def read_vocab2():
    with open('vocab.txt', encoding='utf8') as file:
        vocabulary_list = [k.strip() for k in file.readlines()]
    id2word_dict = dict([(a, b) for a, b in enumerate(vocabulary_list)])
    print('id2word_dict===', id2word_dict)
    # id2word_dict == = {0: 'PAD', 1: 'UNK', 2: 'GO', 3: 'EOS', 4: ',', 5: '我', 6:
    # 写进对象
    dbfile = open('id_to_word_obj', 'wb')
    pickle.dump(id2word_dict, dbfile)
    return  id2word_dict
def convert_to_vector_ask():
    dbfile = open('word_to_id_obj', 'rb')
    word_to_id_obj = pickle.load(dbfile)
    unk_id = word_to_id_obj.get('UNK')
    print('unk_id',unk_id)
    all_ask_vec=[]
    with open('ask_convs.txt', 'r',encoding='utf-8') as f:
        for line in f:
            one_ask=[]
            for words in line.strip():
                one_ask.append(word_to_id_obj.get(words, unk_id))
            all_ask_vec.append(one_ask)
    with  open('all_ask_vec.txt', mode='w', encoding='utf-8') as f:
        for textvec in all_ask_vec:
            f.write(" ".join([str(num) for num in textvec]) + "\n")
            # 写进对象
    dbfile = open('all_ask_vec_obj', 'wb')  # 必须以2进制打开文件，否则pickle无法将对象序列化只文件
    pickle.dump(all_ask_vec, dbfile)

def convert_to_vector_ask2():
    dbfile = open('word_to_id_obj', 'rb')
    word_to_id_obj = pickle.load(dbfile)
    unk_id = word_to_id_obj.get('UNK')
    print('unk_id', unk_id)
    all_ask_vec=[]
    with open('ask_convs2.txt', 'r',encoding='utf-8') as f:
        for line in f:
            one_ask=[]
            for words in line.strip():
                one_ask.append(word_to_id_obj.get(words, unk_id))#选择关系，字典中有就是words,没有，就是UNK_ID
            all_ask_vec.append(one_ask)
def convert_to_vector_response():
    dbfile = open('word_to_id_obj', 'rb')
    word_to_id_obj = pickle.load(dbfile)
    unk_id = word_to_id_obj.get('UNK')
    print('unk_id', unk_id)
    all_ask_vec=[]
    with open('response_convs.txt', 'r',encoding='utf-8') as f:
        for line in f:
            one_ask=[]
            for words in line.strip():
                one_ask.append(word_to_id_obj.get(words, unk_id))
            all_ask_vec.append(one_ask)
    with  open('all_response_vec.txt', mode='w', encoding='utf-8') as f:
        for textvec in all_ask_vec:
            f.write(" ".join([str(num) for num in textvec]) + "\n")
    dbfile = open('all_response_vec_obj', 'wb')  # 必须以2进制打开文件，否则pickle无法将对象序列化只文件
    pickle.dump(all_ask_vec, dbfile)

'数据集打散'
def batch_iter():
    dbfile = open('all_ask_vec_obj', 'rb')
    all_ask_vec2 = pickle.load(dbfile)
    dbfile = open('all_response_vec_obj', 'rb')
    all_response_vec2 = pickle.load(dbfile)
    all_ask_vec=np.array(all_ask_vec2)
    all_response_vec=np.array(all_response_vec2)
    """生成批次数据"""
    data_len = len(all_ask_vec2)
    print('data_len====', data_len)
    indices = np.random.permutation(np.arange(data_len))
    all_ask_vec_shuffle = all_ask_vec[indices]
    all_response_vec_shuffle = all_response_vec[indices]
    dbfile = open('all_ask_vec_shuffle_obj', 'wb')
    pickle.dump(all_ask_vec_shuffle, dbfile)
    dbfile.close()
    dbfile = open('all_response_vec_shuffle_obj', 'wb')
    pickle.dump(all_response_vec_shuffle, dbfile)
    dbfile.close()
def look_text_detail():
    with open("ask_convs.txt", "r", encoding="utf-8") as f:
        source_text = f.read()

    # French target data
    with open("response_convs.txt", "r", encoding="utf-8") as f:
        target_text = f.read()
    sentences = source_text.split('\n')
    print('len(sentences)==\n', len(sentences))
    all_ask_len=0
    source_max_len=0
    for sen in sentences:
        if len(sen)>source_max_len:
            source_max_len=len(sen)
        all_ask_len+=len(sen)
    print('source_max_len==\n', source_max_len)
    print('all_ask_len==\n', all_ask_len)
    print('ave_all_ask_len==\n', all_ask_len/len(sentences))
    sentences = target_text.split('\n')
    print('len(sentences)==\n', len(sentences))
    all_target_len = 0
    target_max_len=0
    for sen in sentences:
        if len(sen)>target_max_len:
            target_max_len=len(sen)
        all_target_len += len(sen)
    print('target_max_len==\n', target_max_len)
    print('all_target_len==\n', all_target_len)
    print('ave_all_ask_len==\n', all_target_len / len(sentences))

# len(sentences)==
#  1546631
# source_max_len==
#  98
# all_ask_len==
#  14687393
# ave_all_ask_len==
#  9.496378256998598
# len(sentences)==
#  1546631
# target_max_len==
#  116
# all_target_len==
#  14694506
# ave_all_ask_len==
#  9.500977285467574
def source_text_to_index():
    max_length = 98
    dbfile = open('word_to_id_obj', 'rb')
    word_to_id_obj = pickle.load(dbfile)
    pad_id = word_to_id_obj.get('PAD')
    print('pad_id===', pad_id)
    dbfile = open('all_ask_vec_shuffle_obj', 'rb')
    all_ask_vec_shuffle = pickle.load(dbfile)
    all_ask_vec_shuffle_pad=[]
    print('all_ask_vec_shuffle==', all_ask_vec_shuffle)
    for one_ask_vec2 in all_ask_vec_shuffle:
        if len(one_ask_vec2) > max_length:
            one_ask_vec=one_ask_vec2[:max_length]
        else:
            one_ask_vec = one_ask_vec2 + [pad_id] * (max_length - len(one_ask_vec2))
        all_ask_vec_shuffle_pad.append(one_ask_vec)
    all_ask_vec_shuffle_pad2=np.array(all_ask_vec_shuffle_pad)
    dbfile = open('all_ask_vec_shuffle_pad_obj', 'wb')  # 必须以2进制打开文件，否则pickle无法将对象序列化只文件
    pickle.dump(all_ask_vec_shuffle_pad2, dbfile)
    dbfile.close()
def target_text_to_index():
    max_length = 118
    dbfile = open('word_to_id_obj', 'rb')
    word_to_id_obj = pickle.load(dbfile)
    pad_id = word_to_id_obj.get('PAD')
    print('pad_id===', pad_id)
    eos_id = word_to_id_obj.get('EOS')
    print('eos_id===', eos_id)

    dbfile = open('all_response_vec_shuffle_obj', 'rb')
    all_response_vec_shuffle = pickle.load(dbfile)
    all_response_vec_shuffle_pad=[]
    print('all_response_vec_shuffle==', all_response_vec_shuffle)
    for one_response_vec2 in all_response_vec_shuffle:
        one_response_vec2.append(eos_id)
        if len(one_response_vec2) > max_length:
            one_response_vec=one_response_vec2[:max_length]
        else:
            one_response_vec = one_response_vec2 + [pad_id] * (max_length - len(one_response_vec2))
        all_response_vec_shuffle_pad.append(one_response_vec)
    all_response_vec_shuffle_pad2=np.array(all_response_vec_shuffle_pad)
    dbfile = open('all_response_vec_shuffle_pad_obj', 'wb')
    pickle.dump(all_response_vec_shuffle_pad2, dbfile)
    dbfile.close()

'数据集合划分训练集合，测试集合，验证集合'
def split_data():
    train_set=0.65
    val_set=0.15
    test_set=0.2
    dbfile = open('all_ask_vec_shuffle_pad_obj', 'rb')
    all_ask_vec_shuffle = pickle.load(dbfile)
    dbfile = open('all_response_vec_shuffle_pad_obj', 'rb')
    all_response_vec_shuffle = pickle.load(dbfile)
    data_len = len(all_ask_vec_shuffle)
    print('data_len====', data_len)
    '-----------------------------------------------------------------------------------------------------------'
    all_ask_vec_shuffle_train=all_ask_vec_shuffle[:int(data_len-data_len*val_set-data_len*test_set)]
    all_ask_vec_shuffle_val = all_ask_vec_shuffle[int(data_len-data_len*val_set-data_len*test_set):int(data_len - data_len * test_set)]
    all_ask_vec_shuffle_test = all_ask_vec_shuffle[int(data_len - data_len * test_set):data_len]
    # 写进对象
    dbfile = open('all_ask_vec_shuffle_train_obj', 'wb')
    pickle.dump(all_ask_vec_shuffle_train, dbfile)
    dbfile.close()
    # 写进对象
    dbfile = open('all_ask_vec_shuffle_val_obj', 'wb')
    pickle.dump(all_ask_vec_shuffle_val, dbfile)
    dbfile.close()
    # 写进对象
    dbfile = open('all_ask_vec_shuffle_test_obj', 'wb')
    pickle.dump(all_ask_vec_shuffle_test, dbfile)
    dbfile.close()
    print('len(all_ask_vec_shuffle_train)====', len(all_ask_vec_shuffle_train))
    print('len(all_ask_vec_shuffle_val)====', len(all_ask_vec_shuffle_val))
    print('len(all_ask_vec_shuffle_test)====', len(all_ask_vec_shuffle_test))
    '------------------------------------------------------------------------------------------'
    all_response_vec_shuffle_train = all_response_vec_shuffle[:int(data_len - data_len * val_set - data_len * test_set)]
    all_response_vec_shuffle_val = all_response_vec_shuffle[int(data_len - data_len * val_set - data_len * test_set):int(
        data_len - data_len * test_set)]
    all_response_vec_shuffle_test = all_response_vec_shuffle[int(data_len - data_len * test_set):data_len]
    # 写进对象
    dbfile = open('all_response_vec_shuffle_train_obj', 'wb')
    pickle.dump(all_response_vec_shuffle_train, dbfile)
    dbfile.close()
    # 写进对象
    dbfile = open('all_response_vec_shuffle_val_obj', 'wb')
    pickle.dump(all_response_vec_shuffle_val, dbfile)
    dbfile.close()
    # 写进对象
    dbfile = open('all_response_vec_shuffle_test_obj', 'wb')
    pickle.dump(all_response_vec_shuffle_test, dbfile)
    dbfile.close()
    print('len(all_response_vec_shuffle_train)====', len(all_response_vec_shuffle_train))
    print('len(all_response_vec_shuffle_val)====', len(all_response_vec_shuffle_val))
    print('len(all_response_vec_shuffle_test)====', len(all_response_vec_shuffle_test))
    # len(all_ask_vec_shuffle_train)==== 1005309
    # len(all_ask_vec_shuffle_val)==== 231995
    # len(all_ask_vec_shuffle_test)==== 309326
    # len(all_response_vec_shuffle_train)==== 1005309
    # len(all_response_vec_shuffle_val)==== 231995
    # len(all_response_vec_shuffle_test)==== 309326
    '--------------------------------------------------------------------------------------------------------'



if __name__=='__main__':
    split_data()

