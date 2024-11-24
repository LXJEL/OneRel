from torch import nn
from transformers import *
import torch


class RelModel(nn.Module):
    def __init__(self, config):
        super(RelModel, self).__init__()
        self.config = config
        self.bert_dim = config.bert_dim
        # self.bert_encoder = BertModel.from_pretrained("..bert-base-chinse", cache_dir='./pre_trained_bert')
        self.bert_encoder = BertModel.from_pretrained("./pre_trained_bert/")
        # 定义一个线性层，该层的输入维度为self.bert_dim * 3，输出维度为self.config.rel_num * self.config.tag_size。
        self.relation_matrix = nn.Linear(self.bert_dim * 3, self.config.rel_num * self.config.tag_size)
        # 定义另一个线性层，该层的输入维度为self.bert_dim * 2，输出维度为self.bert_dim * 3
        self.projection_matrix = nn.Linear(self.bert_dim * 2, self.bert_dim * 3)

        # 定义一个dropout层，dropout率为self.config.dropout_prob
        self.dropout = nn.Dropout(self.config.dropout_prob)
        # 同样定义一个dropout层，dropout率为self.config.entity_pair_dropout
        self.dropout_2 = nn.Dropout(self.config.entity_pair_dropout)
        # 定义激活函数
        self.activation = nn.ReLU()

    def get_encoded_text(self, token_ids, mask):
        # 获取给定token_ids和mask的编码文本表示。
        encoded_text = self.bert_encoder(token_ids, attention_mask=mask)[0]
        return encoded_text

    '''
    在函数`triple_score_matrix`中，将`encoded_text`进行拓展和重复以生成对应的实体对的表示，并将这些表示拼接在一起，形成`entity_pairs`。这种操作的目的是为了计算所有可能三元组之间的关系分数。
    通过使用拓展和重复操作，我们可以将每个实体对的表示进行组合，使得每个实体对之间的关系能够进行比较和评估。拼接后的`entity_pairs`矩阵的形状为`[batch_size, seq_len * seq_len, bert_dim*2]`，其中`seq_len`表示输入文本的长度，`bert_dim`表示BERT模型的维度。
    接下来，我们通过应用线性转换和激活函数，将`entity_pairs`映射到一个新的空间中，以获得更适合表示三元组关系的特征表达。通过在后续的关系矩阵操作中，可以计算每个实体对的关系分数。
    这些步骤的目的是为了模型能够在给定输入文本的情况下，对其中所有可能的三元组做出关系预测。通过计算三元组的关系分数并进行适当的变换和处理，可以得到每个实体对在每个关系标签上的概率分布，用于预测最终的关系。这样做的目的是为了使模型能够捕捉到文本中所有可能的实体之间的关系，并进行关系分类和预测。
    '''
    def triple_score_matrix(self, encoded_text, train = True):
        # encoded_text: [batch_size, seq_len, bert_dim(768)] 1,2,3
        batch_size, seq_len, bert_dim = encoded_text.size()

        # 将encoded_text在第2个维度上进行扩展，得到head_representation
        # head: [batch_size, seq_len * seq_len, bert_dim(768)] 1,1,1, 2,2,2, 3,3,3
        head_representation = encoded_text.unsqueeze(2).expand(batch_size, seq_len, seq_len, bert_dim).reshape(batch_size, seq_len*seq_len, bert_dim)
        
        # # 通过重复张量encoded_text来生成tail_representation
        # tail: [batch_size, seq_len * seq_len, bert_dim(768)] 1,2,3, 1,2,3, 1,2,3
        tail_representation = encoded_text.repeat(1, seq_len, 1)

        # 将head_representation和tail_representation在最后一个维度上拼接起来，得到entity_pairs
        # [batch_size, seq_len * seq_len, bert_dim(768)*2]
        entity_pairs = torch.cat([head_representation, tail_representation], dim=-1)

        # 通过投影矩阵对entity_pairs进行线性转换
        # [batch_size, seq_len * seq_len, bert_dim(768)*3]
        entity_pairs = self.projection_matrix(entity_pairs)

        entity_pairs = self.dropout_2(entity_pairs)

        entity_pairs = self.activation(entity_pairs)

        # [batch_size, seq_len * seq_len, rel_num * tag_size] -> [batch_size, seq_len, seq_len, rel_num, tag_size]
        triple_scores = self.relation_matrix(entity_pairs).reshape(batch_size, seq_len, seq_len, self.config.rel_num, self.config.tag_size)
        
        if train:
            # [batch_size, tag_size, rel_num, seq_len, seq_len]
            return triple_scores.permute(0,4,3,1,2)
        else:
            # [batch_size, seq_len, seq_len, rel_num]
            return triple_scores.argmax(dim = -1).permute(0,3,1,2)
        '''
        在函数 triple_score_matrix 中， triple_scores 的维度为 [batch_size, seq_len, seq_len, rel_num, tag_size]。 在返回结果之前，根据条件 train 的值进行不同的变换。

        当 train=True 时，通过 triple_scores.permute(0,4,3,1,2) 需要改变维度的顺序，详细解释如下：

        0:维度 0,维持不变,表示批次大小。
        4:维度 4 移到第 1 个位置,表示标签的数量(tag_size)。
        3:维度 3 移到第 2 个位置,表示关系的数量(rel_num)。
        1:维度 1 移到第 3 个位置,表示实体对的第一个序列长度(seq_len)。
        2:维度 2 移到第 4 个位置,表示实体对的第二个序列长度(seq_len)。
        因此，操作后的结果的维度为 [batch_size, tag_size, rel_num, seq_len, seq_len]。这种变换的目的是为了使结果的维度能够与标签和关系相对应，以便于训练模型时的计算和损失计算。

        当 train=False 时，使用 triple_scores.argmax(dim=-1).permute(0,3,1,2) 获取预测结果。该操作的含义如下：

        0:维度 0,维持不变,表示批次大小。
        3:维度 3 移到第 1 个位置,表示关系的数量(rel_num)。
        1:维度 1 移到第 2 个位置,表示实体对的第一个序列长度(seq_len)。
        2:维度 2 移到第 3 个位置,表示实体对的第二个序列长度(seq_len)。
        通过 argmax(dim=-1),在最后一个维度上求取每个实体对对应的关系标签的最大值，得到预测结果。最终，通过 permute,将维度顺序调整为 [batch_size, seq_len, seq_len, rel_num]，以便于后续的分析和使用。
        '''


    def forward(self, data, train = True):
        # [batch_size, seq_len]
        token_ids = data['token_ids']
        # [batch_size, seq_len]
        mask = data['mask']
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.get_encoded_text(token_ids, mask)
        # 1 34 768
        encoded_text = self.dropout(encoded_text)

        # [batch_size, rel_num, seq_len, seq_len]
        output = self.triple_score_matrix(encoded_text, train)

        return output
