import pandas as pd
import re
from clean_gadget import clean_gadget
from lang_processors.cpp_processor import CppProcessor
import csv
import time
import numpy as np

def normalization(source):
    """
    normalization code
    :param source: dataframe
    :return:
    """
    cpp_processor = CppProcessor()
    nor_code = []
    for fun in source['code']:
        lines = fun.split('\n')
        # print(lines)
        code = ''
        for line in lines:
            line = line.strip()
            line = re.sub('//.*', '', line)
            code += line + ' '
        # code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code)
        code = re.sub('/\\*.*?\\*/', '', code)
        code = clean_gadget([code])
        code[0] = re.sub('"".*?""', '', code[0], 20)
        code_list = cpp_processor.tokenize_code(code[0])
        print(len(code_list))

        tokenization_code = ''
        for token in code_list:
            tokenization_code = tokenization_code + token + " "
        nor_code.append(tokenization_code)
        # print(tokenization_code)
    return nor_code


def normalization2(source):
    cpp_processor = CppProcessor()
    nor_code = []
    for fun in source['code']:
        lines = fun.split('\n')
        # print(lines)
        code = ''
        for line in lines:
            line = line.strip()
            line = re.sub('//.*', '', line)
            code += line + ' '
        # code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code)
        code = re.sub('/\\*.*?\\*/', '', code)
        code = clean_gadget([code])
        code[0] = re.sub('"".*?""', '', code[0], 20)

        code_list = cpp_processor.tokenize_code(code[0])
        # nor_code.append(code[0])
        # nor_code.append(code6)
        tokenization_code = ''
        for token in code_list:
            tokenization_code = tokenization_code + token + " "
        nor_code.append(tokenization_code)
        # print(tokenization_code)
        with open('./corpus.txt', 'a') as f:
            f.write(tokenization_code)
            f.write('\n')
    return nor_code


def mutrvd():
    train = pd.read_pickle('trvd_train.pkl')
    test = pd.read_pickle('trvd_test.pkl')
    val = pd.read_pickle('trvd_val.pkl')

    train['code'] = normalization(train)
    train.to_pickle('./mutrvd/train.pkl')

    test['code'] = normalization(test)
    test.to_pickle('./mutrvd/test.pkl')

    val['code'] = normalization(val)
    val.to_pickle('./mutrvd/val.pkl')


def nor(source):
    cpp_processor = CppProcessor()
    nor_code = []
    lines = source.split('\n')
    # print(lines)
    code = ''
    for line in lines:
        line = line.strip()
        line = re.sub('//.*', '', line)
        code += line + ' '
    # code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code)
    code = re.sub('/\\*.*?\\*/', '', code)
    code = clean_gadget([code])
    # code[0] = code[0].replace('"".*?""', '', 10)
    code[0] = re.sub('"".*?""', '', code[0], 20)

    code_list = cpp_processor.tokenize_code(code[0])
    tokenization_code = ''
    for token in code_list:
        tokenization_code = tokenization_code + token + " "
    nor_code.append(tokenization_code)
    # print(tokenization_code)
    return nor_code


if __name__ == '__main__':
    start_time = time.time()
    code_train = pd.read_csv('../dataset/trvd_test/our_train.csv')
    data = pd.DataFrame(code_train)
    list = []
    for i in range(len(data)):
        s = data['text'][i]
        t = nor(s)
        t=' '.join(t)
        list.append(t)
    rows = zip(list)
    with open('../dataset/trvd_test/our_line.csv', "a") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    end_time = time.time()
    execution_time = end_time - start_time
    print("time:", execution_time, "seconds")
