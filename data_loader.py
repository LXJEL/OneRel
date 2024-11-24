from torch.utils.data import DataLoader, Dataset
import json
import os
import torch
from utils import get_tokenizer
from utils.tokenization import BasicTokenizer
import numpy as np
from random import choice
from transformers import BertTokenizer

tokenizer = get_tokenizer('pre_trained_bert/vocab.txt')
# tokenizer = BertTokenizer.from_pretrained('pre_trained_bert/vocab.txt')
basicTokenizer = BasicTokenizer(do_lower_case=False)
tag_file = 'data/tag.txt' 

def find_head_idx(source, target):
    # 在tokens中找头、尾实体
    # 若存在则返回头/尾实体开始的位置
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1

# 完成自定义数据集，新建REDataset类继承torch的Dataset类
class REDataset(Dataset):
    def __init__(self, config, prefix, is_test, tokenizer):
        self.config = config
        # prefix就是传过来的文件名称
        self.prefix = prefix
        self.is_test = is_test
        self.tokenizer = tokenizer

        # 首先读取json文件的数据存储到self.json_data中
        if self.config.debug:
            self.json_data = json.load(open(os.path.join(self.config.data_path, prefix + '.json')))[:12]
        else:
            self.json_data = json.load(open(os.path.join(self.config.data_path, prefix + '.json')))
        
        # 读取关系到id的对应，存到self.rel2id中
        self.rel2id = json.load(open(os.path.join(self.config.data_path, 'rel2id.json')))[1]
        # 读取tag到id的对应，存到self.tag2id中，为关系矩阵的标记做准备     {"A": 0,"HB-TB": 1,"HB-TE": 2,"HE-TE": 3}
        self.tag2id = json.load(open('data/tag2id.json'))[1]

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        # __getitem__函数接受一个索引idx作为输入，并获取与该索引相关联的JSON数据。然后从JSON数据中提取文本信息。
        ins_json_data = self.json_data[idx]
        text = ins_json_data['text']
        # text = ' '.join(text.split()[:self.config.max_len])
        # text = basicTokenizer.tokenize(text)
        # text = " ".join(text[:self.config.max_len])
        # 使用分词器对文本进行分词。如果分词后的token数超过了在config中定义的最大长度，就进行截断。变量text_len存储了分词后的文本长度。
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.config.bert_max_len:
            tokens = tokens[: self.config.bert_max_len]
        # 对text进行分词后都存在tokens中
        text_len = len(tokens)

        # 训练
        if not self.is_test:
            # 如果代码不处于测试模式，它会创建一个字典s2ro_map来存subject-object关系映射。 map[subject] = object+relation编号
            # 它遍历JSON数据中的triple_list，对主题和客体字符串进行分词，找到它们在分词后的文本中的相应索引，并将它们存储在s2ro_map字典中。
            s2ro_map = {}
            for triple in ins_json_data['triple_list']:
                # 0和2为头尾实体，1为关系
                triple = (self.tokenizer.tokenize(triple[0])[1:-1],
                         triple[1], self.tokenizer.tokenize(triple[2])[1:-1])
                sub_head_idx = find_head_idx(tokens, triple[0])
                obj_head_idx = find_head_idx(tokens, triple[2])
                # 若头尾实体均可以在tokens中找到
                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                    if sub not in s2ro_map:
                        s2ro_map[sub] = []
                    s2ro_map[sub].append((obj_head_idx, obj_head_idx + len(triple[2]) - 1, self.rel2id[triple[1]]))
            
            if s2ro_map:
                # token_ids：一个整数列表，表示经过编码的文本的标记或令牌。
                # segment_ids：一个整数列表，用于指示文本的不同部分，例如句子的开始和结束。在一些语言模型中，这部分可能用于表示上下文的分段。
                token_ids, segment_ids = self.tokenizer.encode(first=text)
                # masks 作为掩码用来指示输入文本的哪些部分是真实的文本内容，哪些部分是填充或无效的。通过将 segment_ids 赋值给 masks，可以将 segment_ids 中的信息直接应用于掩码。
                # 通常，segment_ids 用于处理输入中的不同句子或段落，其中每个句子或段落都被分配一个唯一的 segment_id。在这种情况下，masks 在处理文本时可以通过将填充部分标记为 0 或无效来忽略它们。
                masks = segment_ids
                if len(token_ids) > text_len:
                    token_ids = token_ids[:text_len]
                    masks = masks[:text_len]
                mask_length = len(masks)
                # 将token_ids和masks转化为numpy数组，对masks中的每个元素加1
                token_ids = np.array(token_ids)
                masks = np.array(masks) + 1
                # 创建一个shape为(mask_length, mask_length)的全1数组loss_masks。
                loss_masks = np.ones((mask_length, mask_length))
                # 创建一个shape为(self.config.rel_num, text_len, text_len)、元素都为0的triple_matrix数组
                triple_matrix = np.zeros((self.config.rel_num, text_len, text_len))

                for s in s2ro_map:
                    sub_head = s[0]
                    sub_tail = s[1]
                    for ro in s2ro_map.get((sub_head, sub_tail), []):
                        obj_head, obj_tail, relation = ro
                        triple_matrix[relation][sub_head][obj_head] = self.tag2id['HB-TB']
                        triple_matrix[relation][sub_head][obj_tail] = self.tag2id['HB-TE']
                        triple_matrix[relation][sub_tail][obj_tail] = self.tag2id['HE-TE']

                return token_ids, masks, loss_masks, text_len, triple_matrix, ins_json_data['triple_list'], tokens
            else:
                print(ins_json_data)
                return None

        # 测试
        else:
            token_ids, masks = self.tokenizer.encode(first=text)
            if len(token_ids) > text_len:
                token_ids = token_ids[:text_len]
                masks = masks[:text_len]
            token_ids = np.array(token_ids)
            masks = np.array(masks) + 1
            mask_length = len(masks)
            loss_masks = np.array(masks) + 1
            triple_matrix = np.zeros((self.config.rel_num, text_len, text_len))
            return token_ids, masks, loss_masks, text_len, triple_matrix, ins_json_data['triple_list'], tokens

def re_collate_fn(batch):
    # 在数据加载过程中对批次数据进行重新组织和处理
    # 过滤掉值为None的样本
    batch = list(filter(lambda x: x is not None, batch))

    # 根据文本长度降序对批次数据进行排序
    batch.sort(key=lambda x: x[3], reverse=True)
    
    # 拆分批次数据
    token_ids, masks, loss_masks, text_len, triple_matrix, triples, tokens = zip(*batch)
    cur_batch_len = len(batch)
    max_text_len = max(text_len)

    # 初始化张量用于存储批次数据
    batch_token_ids = torch.LongTensor(cur_batch_len, max_text_len).zero_()
    batch_masks = torch.LongTensor(cur_batch_len, max_text_len).zero_()
    batch_loss_masks = torch.LongTensor(cur_batch_len, 1, max_text_len, max_text_len).zero_()
    # if use WebNLG_star, modify 24 to 171
    # if use duie, modify 24 to 48
    # batch_triple_matrix = torch.LongTensor(cur_batch_len, 24, max_text_len, max_text_len).zero_()
    batch_triple_matrix = torch.LongTensor(cur_batch_len, 4, max_text_len, max_text_len).zero_()

    # 将数据拷贝到对应的张量中
    for i in range(cur_batch_len):
        batch_token_ids[i, :text_len[i]].copy_(torch.from_numpy(token_ids[i]))
        batch_masks[i, :text_len[i]].copy_(torch.from_numpy(masks[i]))
        batch_loss_masks[i, 0, :text_len[i], :text_len[i]].copy_(torch.from_numpy(loss_masks[i]))
        batch_triple_matrix[i, :, :text_len[i], :text_len[i]].copy_(torch.from_numpy(triple_matrix[i]))

    # 返回重新组织后的批次数据
    return {'token_ids': batch_token_ids,
            'mask': batch_masks,
            'loss_mask': batch_loss_masks,
            'triple_matrix': batch_triple_matrix,
            'triples': triples,
            'tokens': tokens}

def get_loader(config, prefix, is_test=False, num_workers=0, collate_fn=re_collate_fn):
    dataset = REDataset(config, prefix, is_test, tokenizer)
    if not is_test:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    else:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    return data_loader

class DataPreFetcher(object):
    def __init__(self, loader):
        # 将数据加载器转换为一个迭代器self.loader
        self.loader = iter(loader)
        # 使用CUDA流
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            for k, v in self.next_data.items():
                if isinstance(v, torch.Tensor):
                    self.next_data[k] = self.next_data[k].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data

