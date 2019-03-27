# _*_ coding:utf-8 _*_

import os
import jieba
import sys

train_file = './data/cnews.train.txt'
test_file = './data/cnews.test.txt'
val_file = './data/cnews.val.txt'

seg_train_file = './data/cnews.train.seg.txt'
seg_test_file = './data/cnews.test.seg.txt'
seg_val_file = './data/cnews.val.seg.txt'

vocab_file = './data/cnews.vocab.txt'
category_file = './data/cnews.category.txt'


def generate_seg_file(input_file, output_seg_file):
    with open(input_file, encoding='utf-8') as f:
        lines = f.readlines()
    with open(output_seg_file, 'wb') as f:
        for line in lines:
            label, content = line.strip('\r\n').split('\t')
            word_iter = jieba.cut(content)
            word_content = ''
            for word in word_iter:
                word = word.strip(' ')
                if word != '':
                    word_content += word + ' '
            out_line = '%s\t%s\n' % (label, word_content.strip(' '))
            f.write(out_line.encode('utf-8'))


# generate_seg_file(val_file,seg_val_file)
# generate_seg_file(train_file,seg_train_file)
# generate_seg_file(test_file,seg_test_file)

def generate_vocab_file(input_seg_file, output_vocab_file):
    with open(input_seg_file, 'rb') as f:
        lines = f.readlines()
    word_dict = {}
    for line in lines:
        line = line.decode()
        label, content = line.strip('\r\n').split('\t')
        for word in content.split():
            word_dict.setdefault(word, 0)
            word_dict[word] += 1
    sorted_word_dict = sorted(word_dict.items(), key=lambda d: d[1], reverse=True)
    with open(output_vocab_file, 'wb') as f:
        f.write('<UNK>\t10000000\n'.encode())
        for item in sorted_word_dict:
            str = '%s\t%d\n' % (item[0], item[1])
            f.write(str.encode())


# generate_vocab_file(seg_val_file, vocab_file)
# generate_vocab_file(seg_train_file,val_file)
# generate_vocab_file(seg_test_file,val_file)

def generate_category_dict(input_file, category_file):
    with open(input_file, 'rb') as f:
        lines = f.readlines()
    category_dict = {}
    for line in lines:
        line =line.decode()
        label, content = line.strip('\r\n').split('\t')
        category_dict.setdefault(label,0)
        category_dict[label]+=1
    category_number = len(category_dict)
    with open(category_file,'wb') as f:
        for category in category_dict:
            line = '%s\n' % (category)
            print( '%s\t%d' % (category, category_dict[category]))
            f.write(line.encode('utf-8'))

generate_category_dict(train_file, category_file)
