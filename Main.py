#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@software: PyCharm Community Edition
@file: Main.py
@time: 1/7/19 10:58 AM
"""
import fasttext


if __name__ == "__main__":
    # classifier = fasttext.supervised('cooking.train', 'model_cooking', dim=64, lr=0.5, label_prefix='__label__', epoch=200)
    classifier = fasttext.load_model("model_cooking.bin", label_prefix="__label__")
    result = classifier.test('cooking.train')
    print('P@1:', result.precision)
    print('R@1:', result.recall)
    print('Number of train examples:', result.nexamples)

    result = classifier.test('cooking.valid')
    print('P@1:', result.precision)
    print('R@1:', result.recall)
    print('Number of valid examples:', result.nexamples)

    sents = ['How to make edges on pancakes be crispy and the inside soft', 'Roasting sirloin which has already been cut into slices']
    classifier_result = classifier.predict(sents, k=3)
    print(classifier_result)