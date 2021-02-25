import pandas as pd
import numpy as np


data_dict = {}
new_data_dict = {}
words = []
sentence = []
sentence_2 = []
final_input_matrix = []
final_input_matrix_2 = []

datas = pd.read_csv('train.csv')
text = datas['text']
category = datas['Category']

datas_2 = pd.read_csv('test.csv')
id = datas_2['Id']
text_2 = datas_2['text']
category_2 = datas_2['Category']


def set_data():
    for i in range(len(text)):
        test = text[i]
        test = test.split(' ')
        for j in test:
            words.append(j)
    for word in words:
        if word not in data_dict:
            data_dict[word] = 1
        data_dict[word] += 1
    return data_dict


def improve_data(a):
    for _ in data_dict:
        num = data_dict.get(_)
        # if 10 < num < 10000:
        if 10 < num < 50000:
            if _ not in new_data_dict:
                new_data_dict[_] = 1
            else:
                pass
    return new_data_dict, len(new_data_dict)


def reset(x):
    for _ in new_data_dict:
        new_data_dict[_] = 0
    return new_data_dict


def deal_with_training_input_data():
    for k in range(len(text)):
        new_test = text[k]
        new_test = new_test.split(' ')
        sentence.append(new_test)

    for L in range(len(sentence)):
        word = sentence[L]
        for m in range(len(word)):
            word = sentence[L][m]
            if word not in new_data_dict:
                pass
            else:
                new_data_dict[word] += 1
        final_input_matrix.append(list(new_data_dict.values()))
        reset(new_data_dict)
    arr = np.array(final_input_matrix)

    return arr


def deal_with_training_output_data():
    training_output = np.array(category)
    return training_output


def deal_with_testing_input_data():
    for k in range(len(text_2)):
        new_test = text_2[k]
        new_test = new_test.split(' ')
        sentence_2.append(new_test)
    for L in range(len(sentence_2)):
        word = sentence_2[L]
        for m in range(len(word)):
            word = sentence_2[L][m]
            if word not in new_data_dict:
                pass
            else:
                new_data_dict[word] += 1
        final_input_matrix_2.append(list(new_data_dict.values()))
        reset(new_data_dict)
    arr = np.array(final_input_matrix_2)
    return arr


def deal_with_testing_output_data():
    testing_output = np.array(category_2)
    return testing_output


def progress_input_data():
    data = set_data()
    data = improve_data(data)
    data = reset(data)
    feature = len(data)
    training_input = deal_with_training_input_data()
    testing_input = deal_with_testing_input_data()
    return training_input, testing_input, feature


# def upload_data():
#     predict_answer = pytorch_network.overwrite_data(pytorch_network.test_x)
#     mid_term_marks = {"Id": id,
#                       "Category": predict_answer, }
#     mid_term_marks_df = pd.DataFrame(mid_term_marks)
#     mid_term_marks_df.to_csv("new_sample_submission.csv")


if __name__ == '__main__':
    upload_data()


