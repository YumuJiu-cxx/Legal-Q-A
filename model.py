import pandas as pd
import json
import os
import random
import re
import jieba
import pickle
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Flatten
from keras.layers import concatenate
from keras.utils import to_categorical
from keras.utils import pad_sequences

# 读取json文件
folder_path = 'train'
files = os.listdir(folder_path)
data = []
for file in files:
    with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
        data.append(json.load(f))

# 预处理数据
questions = []
answers = []
labels = []
for d in data:
    correct_answer = d['answer']
    d['candidate_answer'].append(correct_answer)
    random.shuffle(d['candidate_answer'])  # 打乱候选答案的顺序
    for a in d['candidate_answer']:
        questions.append(d['question'])
        answers.append(a)
        labels.append(1 if a == correct_answer else 0)

# 构建DataFrame
dist = {"questions": questions,
        "answers": answers,
        "labels": labels}
df = pd.DataFrame(dist)


# 存储DataFrame
# df.to_csv('processed_data.csv', index=False, encoding='utf-8')

def stopwordslist(filepath):
    """删除停用词"""
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]

    return stopwords


# 加载停用词
stopwords = stopwordslist("Data/chineseStopWords.txt")


def remove_punctuation(line):
    """删除除字母、数字、汉字以外的所有符号"""
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)

    return line


# 删除除字母、数字、汉字以外的所有符号
df['questions'] = df['questions'].apply(remove_punctuation)
# 分词，并过滤停用词
df['questions'] = df['questions'].apply(lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stopwords]))

df['answers'] = df['answers'].apply(remove_punctuation)
df['answers'] = df['answers'].apply(lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stopwords]))

"""
  LSTM建模
"""
# 定义常量
max_nb_words = 50000  # 设置最频繁使用的50000个词
max_sequence_length = 250  # 设置每条df['clean_review']最大的长度
embedding_dim = 100  # 设置Embedding层的维度

tokenizer = Tokenizer(num_words=max_nb_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['questions'].values)
word_index = tokenizer.word_index
print('共有 %s 个不相同的词语.' % len(word_index), '\n')

# 保存tokenizer
with open('Tokenizer/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 转化为序列
sequences_questions = tokenizer.texts_to_sequences(df['questions'].values)
sequences_answers = tokenizer.texts_to_sequences(df['answers'].values)

# Padding
padded_questions = pad_sequences(sequences_questions, maxlen=max_sequence_length)
padded_answers = pad_sequences(sequences_answers, maxlen=max_sequence_length)

# 划分训练集和测试集
X_train_question, X_test_question, X_train_answer, X_test_answer, y_train, y_test = train_test_split(padded_questions,
                                                                                                     padded_answers, df[
                                                                                                         'labels'].values,
                                                                                                     test_size=0.01)

# 定义模型
question_input = Input(shape=(max_sequence_length,), dtype='int32', name='question_input')
embedded_question = Embedding(len(word_index) + 1, embedding_dim, input_length=max_sequence_length)(question_input)
question_lstm = LSTM(256)(embedded_question)

answer_input = Input(shape=(max_sequence_length,), dtype='int32', name='answer_input')
embedded_answer = Embedding(len(word_index) + 1, embedding_dim, input_length=max_sequence_length)(answer_input)
answer_lstm = LSTM(256)(embedded_answer)

concatenated = concatenate([question_lstm, answer_lstm])
predictions = Dense(1, activation='sigmoid')(concatenated)

model = Model([question_input, answer_input], predictions)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit([X_train_question, X_train_answer], y_train, epochs=10, batch_size=128,
          validation_data=([X_test_question, X_test_answer], y_test))

# 保存模型
model.save('lstm_model.h5')
