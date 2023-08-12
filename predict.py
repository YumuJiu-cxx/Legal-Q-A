from keras import models
import jieba
import json
import os
import re
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# 读取json文件
folder_path = 'test'
files = os.listdir(folder_path)
data = []
for file in files:
    with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
        data.append(json.load(f))

# 预处理数据
questions = []
candidate_answers = []
for d in data:
    questions.append(d['question'])
    candidate_answers.append(d['candidate_answer'])

# 加载模型
model = models.load_model('lstm_model.h5')

# 加载tokenizer
with open('Tokenizer/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def stop_words_list(filepath):
    """删除停用词"""
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]

    return stopwords


# 加载停用词
stopwords = stop_words_list("Data/chineseStopWords.txt")


# 定义删除除字母、数字、汉字以外的所有符号的函数
def remove_punctuation(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line


max_sequence_length = 250
result = []

# 对问题进行预处理
for i in range(len(data)):
    new_question = questions[i]
    new_question = remove_punctuation(new_question)
    new_question = " ".join([w for w in list(jieba.cut(new_question)) if w not in stopwords])
    new_question = tokenizer.texts_to_sequences([new_question])
    new_question = pad_sequences(new_question, maxlen=max_sequence_length)

    # 对每一个备选答案进行预处理和预测
    new_answers = candidate_answers[i]
    predictions = []
    for new_answer in new_answers:
        new_answer = remove_punctuation(new_answer)
        new_answer = " ".join([w for w in list(jieba.cut(new_answer)) if w not in stopwords])
        new_answer = tokenizer.texts_to_sequences([new_answer])
        new_answer = pad_sequences(new_answer, maxlen=max_sequence_length)

        # 使用模型进行预测
        prediction = model.predict([new_question, new_answer])
        predictions.append(prediction)

    # 找出预测分数最高的答案
    best_answer_index = np.argmax(predictions)
    best_answer = new_answers[best_answer_index]

    # 将结果添加到字典中，并将字典添加到结果列表中
    answer_dict = {
        "id": i,
        "question": questions[i],
        "answer": best_answer
    }
    result.append(answer_dict)
    print(answer_dict)

with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
