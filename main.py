import argparse
import os
from tqdm import tqdm
import chardet
import numpy as np
import torch
from transformers import BertTokenizer
from sklearn.metrics import classification_report

from torch.utils.data import Dataset, random_split
from dataset import VistaDataset
from model import VistaNet
from model2 import SentimentClassifier
from train_and_eval import train_with_model, predict, eval_with_ablation


# 通过Bert的index转换将训练集的句子列表转成index,然后用创建的类别字典将数据集的标签转成index
def word_tag_to_index(true_sentences, tokenizer):
    sentences = []
    for sent in true_sentences:
        sent = sent.strip("\n")
        token_sentence = [tokenizer.tokenize(word) for word in sent]
        words = [i for word in token_sentence for i in word]
        words = ['[CLS]'] + words
        sentences.append(tokenizer.convert_tokens_to_ids(words))

    return sentences


def label_to_index(label_list, type2idx):
    label_idx = []
    for label in label_list:
        label_idx.append(type2idx[label])

    return label_idx


def get_data(data_path_root, train_txt_path):
    with open(train_txt_path, 'r') as f:
        txt = f.readlines()
    data_guid_and_label = []
    for i in range(1, len(txt)):
        data_guid_and_label.append(txt[i].strip("\n"))

    data_id_list = []
    text_list = []
    label_list = []

    for data in tqdm(data_guid_and_label):
        idx = data.split(',')
        data_id_list.append(idx[0])
        label_list.append(idx[1])

        with open(data_path_root + idx[0] + '.txt', 'rb') as f:
            r = f.read()
            f_charInfo = chardet.detect(r)
            if f_charInfo['encoding'] is not None:
                text_list.append(r.decode(f_charInfo['encoding'], errors='ignore'))
            else:
                data_id_list.pop()
                label_list.pop()

    return data_id_list, text_list, label_list


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-do_train', type=bool, default=False, help='True:train or Flase:not')
    parse.add_argument('-do_test', type=bool, default=False, help='True:test. tips:during testing, training and verificattion are not supported')
    parse.add_argument('-do_eval_ablation', type=bool, default=False, help='True: eval')
    parse.add_argument('-ablation', type=int, default=0, help='eval type:(0: Bimodal, 1: only text, 2: only image)')
    # 以下为模型和训练过程中使用的参数
    parse.add_argument('-epoch', type=int, default=20, help='train epoch num')
    parse.add_argument('-batch_size', type=int, default=8, help='batch size number')
    parse.add_argument('-lr', type=float, default=1e-3, help='learning rate')
    opt = parse.parse_args()

    type2idx = {'positive': 0, 'neutral': 1, 'negative': 2}
    idx2type = {0: 'positive', 1: 'neutral', 2: 'negative'}

    # 加载Bert
    tokenizer = BertTokenizer.from_pretrained('pretrained_bert_models/bert-base-uncased/', do_lower_case=True)

    data_path_root = './dataset/data/'
    train_txt_path = './dataset/train.txt'
    test_txt_path = './dataset/test_without_label.txt'

    batch_size = opt.batch_size
    model = SentimentClassifier()

    do_train, do_eval_ablation, do_test = True, False, False

    if opt.do_test:
        test_data_id_list, test_text_list, test_label_list = get_data(data_path_root, test_txt_path)

        test_dataset_word_idx = word_tag_to_index(test_text_list, tokenizer)
        test_dataset_label_idx = [0 for i in test_label_list]
        test_dataset = VistaDataset(test_dataset_word_idx, test_data_id_list, test_dataset_label_idx, data_path_root)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                                      collate_fn=test_dataset.collate_fn)
        model_path = "train_model/best-model.pth"
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
        predict(model, test_dataloader)
    else:
        data_id_list, text_list, label_list = get_data(data_path_root, train_txt_path)

        dataset_word_idx = word_tag_to_index(text_list, tokenizer)
        dataset_label_idx = label_to_index(label_list, type2idx)
        dataset = VistaDataset(dataset_word_idx, data_id_list, dataset_label_idx, data_path_root)

        train_dataset, eval_dataset = random_split(dataset, [(len(data_id_list) - 500), 500])

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                       num_workers=0,
                                                       collate_fn=dataset.collate_fn)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                                      collate_fn=dataset.collate_fn)
        if opt.do_train:
            # model = VistaNet(768, len(type2idx))

            train_with_model(model, train_dataloader, eval_dataloader, opt.epoch, opt.lr)
        if opt.do_eval_ablation:
            model_path = "train_model/best-model.pth"
            model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)

            if opt.ablation == 0:
                eval_with_ablation(model, eval_dataloader, torch.nn.CrossEntropyLoss(), 0, 0)
            elif opt.ablation == 1:
                eval_with_ablation(model, eval_dataloader, torch.nn.CrossEntropyLoss(), 0, 1)
            elif opt.ablation == 2:
                eval_with_ablation(model, eval_dataloader, torch.nn.CrossEntropyLoss(), 0, 2)

