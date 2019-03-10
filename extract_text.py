import os
import random
import pickle
'这个是把下载的语料提取出来'
"""
E
M 畹/华/吾/侄/
M 你/接/到/这/封/信/的/时/候/
M 不/知/道/大/伯/还/在/不/在/人/世/了/
E
M 咱/们/梅/家/从/你/爷/爷/起/
M 就/一/直/小/心/翼/翼/地/唱/戏/
M 侍/奉/宫/廷/侍/奉/百/姓/
M 从/来/不/曾/遭/此/大/祸/
M 太/后/的/万/寿/节/谁/敢/不/穿/红/
M 就/你/胆/儿/大/
M 唉/这/我/舅/母/出/殡/
M 我/不/敢/穿/红/啊/
M 唉/呦/唉/呦/爷/
M 您/打/得/好/我/该/打/
M 就/因/为/没/穿/红/让/人/赏/咱/一/纸/枷/锁/
M 爷/您/别/给/我/戴/这/纸/枷/锁/呀/
E
M 您/多/打/我/几/下/不/就/得/了/吗/
M 走/
M 这/是/哪/一/出/啊/…/ / /这/是/
M 撕/破/一/点/就/弄/死/你/
M 唉/
M 记/着/唱/戏/的/再/红/
M 还/是/让/人/瞧/不/起/
M 大/伯/不/想/让/你/挨/了/打/
M 还/得/跟/人/家/说/打/得/好/
M 大/伯/不/想/让/你/再/戴/上/那/纸/枷/锁/
M 畹/华/开/开/门/哪/
E
...
"""
def extract_text():
    convs = []  # 对话集合
    with open('dgk_shooter_z.txt', encoding="utf8") as f:
        one_conv = []  # 一次完整对话
        for line in f:
            line = line.strip('\n').replace('/', '')
            if line == '':
                continue
            if line[0] == 'E':
                if one_conv:
                    convs.append(one_conv)
                one_conv = []
            elif line[0] == 'M':
                one_conv.append(line.split(' ')[1])
                # 写进文件
        convs.append(one_conv)
        with open('all_convs.txt', 'wb') as fW:
            # remove_stopwords_con2, labels = remove_stopwords_con_and_label2()
            for one_conv2 in convs:
                for one_conv_text in one_conv2:
                    fW.write(one_conv_text.encode('utf-8') + ','.encode('utf-8'))
                fW.write('\n'.encode('utf-8'))
            # 写进对象
        dbfile = open('all_convs_obj', 'wb')  # 必须以2进制打开文件，否则pickle无法将对象序列化只文件
        pickle.dump(convs, dbfile)
        dbfile.close()
def extract_text2():
    convs = []  # 对话集合
    with open('dgk_shooter_z2.txt', encoding="utf8") as f:
        one_conv = []  # 一次完整对话
        for line in f:
            line = line.strip('\n').replace('/', '')
            if line == '':
                continue
            if line[0] == 'E':
                if one_conv:
                    convs.append(one_conv)
                one_conv = []
            elif line[0] == 'M':
                one_conv.append(line.split(' ')[1])
                # 写进文件
        convs.append(one_conv)
        with open('all_convs2.txt', 'wb') as fW:
            for one_conv2 in convs:
                for one_conv_text in one_conv2:
                    fW.write(one_conv_text.encode('utf-8') + ','.encode('utf-8'))
                fW.write('\n'.encode('utf-8'))
            # 写进对象
        dbfile = open('all_convs_obj2', 'wb')  # 必须以2进制打开文件，否则pickle无法将对象序列化只文件
        pickle.dump(convs, dbfile)
        dbfile.close()



"""
2222convs=== [['畹华吾侄', '你接到这封信的时候', '不知道大伯还在不在人世了'], ['咱们梅家从你爷爷起'], ['你十三爷爷来看你来了'], ['梅兰芳不来退票'], ['假说公子得中了'], ['十三爷到', '十三爷您吉祥十三爷您硬朗', '托福托福好好', '请请请这边请'], ['22222梅兰芳不来退票']]
len(convs)== 7
one_conv2== ['畹华吾侄', '你接到这封信的时候', '不知道大伯还在不在人世了']
one_conv_text= 畹华吾侄
print(convs[:3])  # 个人感觉对白数据集有点不给力啊
[ ['畹华吾侄', '你接到这封信的时候', '不知道大伯还在不在人世了'], 
  ['咱们梅家从你爷爷起', '就一直小心翼翼地唱戏', '侍奉宫廷侍奉百姓', '从来不曾遭此大祸', '太后的万寿节谁敢不穿红', '就你胆儿大', '唉这我舅母出殡', '我不敢穿红啊', '唉呦唉呦爷', '您打得好我该打', '就因为没穿红让人赏咱一纸枷锁', '爷您别给我戴这纸枷锁呀'], 
  ['您多打我几下不就得了吗', '走', '这是哪一出啊 ', '撕破一点就弄死你', '唉', '记着唱戏的再红', '还是让人瞧不起', '大伯不想让你挨了打', '还得跟人家说打得好', '大伯不想让你再戴上那纸枷锁', '畹华开开门哪'], ....]
"""

# 把对话分成问与答
def ask_response():
    dbfile = open('all_convs_obj', 'rb')
    convs = pickle.load(dbfile)
    ask = []  # 问
    response = []  # 答
    for conv in convs:  #
        # print('conv===', conv)
        if len(conv) == 1:
            continue
        if len(conv) % 2 != 0:  # 奇数对话数, 转为偶数对话
            conv = conv[:-1]
        for i in range(len(conv)):
            # print('i===', i)
            # print('len(conv)===', len(conv))
            if i % 2 == 0:
                ask.append(conv[i])
            else:
                response.append(conv[i])
    with open('ask_convs.txt', 'wb') as fW:
        for ask2 in ask:
            fW.write(ask2.encode('utf-8'))
            fW.write('\n'.encode('utf-8'))
    with open('response_convs.txt', 'wb') as fW:
        for response2 in response:
            fW.write(response2.encode('utf-8'))
            fW.write('\n'.encode('utf-8'))
    # 写进对象
    dbfile = open('ask_convs_obj', 'wb')
    pickle.dump(ask, dbfile)
    dbfile.close()
    # 写进对象
    dbfile = open('response_convs_obj', 'wb')
    pickle.dump(response, dbfile)
    dbfile.close()

if __name__=='__main__':
    extract_text2()

