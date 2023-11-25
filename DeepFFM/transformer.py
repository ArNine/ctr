import math
import pandas as pd
import torch
from torch import nn
# from d2l import torch as d2l
# import study.new_data as d2l


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class EncoderBlock(nn.Module):
    """transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape,ffn_num_inputs, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout,
                                                use_bias)
        self.attention = d2l.MultiHeadAttention(key_size, query_size,value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_inputs, ffn_num_hiddens, num_hiddens
        )
        nn.Transformer
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(d2l.Encoder):
    """transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, use_bias))
    def forward(self, X, valid_lens, *args):
        """因为位置编码值在-1和1之间，因此嵌入值乘以嵌入维度的平方根进行缩放，
        然后再与位置编码相加"""
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

#
# if __name__ =="__main__":
#     ffn = PositionWiseFFN(4, 4, 8)
#     ffn.eval()
#     print("fnn:\n",ffn(torch.ones((2, 3, 4)))[0])
#
#     ln = nn.LayerNorm(2)
#     bn = nn.BatchNorm1d(2)
#     X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
#     # 在训练模式下计算X的均值和方差
#     print('layer norm:', ln(X), '\nbatch norm', bn(X))
#
#     add_norm = AddNorm([3, 4], 0.5)
#     add_norm.eval()
#     print("add_norm: ", add_norm(torch.ones((2, 3, 4)), torch.ones(2, 3, 4)).shape)
#
#
#     X = torch.ones((2, 100, 24))
#     valid_lens = torch.tensor([3, 2])
#     encoder_blk  = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
#     encoder_blk.eval()
#     print("encoder_blk:\n", encoder_blk(X, valid_lens).shape)
#
#     encoder = TransformerEncoder(200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
#     encoder.eval()
#     print("tansformerencoder:\n",encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape)
#
#
#     decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
#     decoder_blk.eval()
#     X = torch.ones((2, 100, 24))
#     state = [encoder_blk(X, valid_lens), valid_lens, [None]]
#     print("decoder_blk:\n",decoder_blk(X, state)[0].shape)
#
#
#     #训练
#     num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
#     lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
#     ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
#     key_size, query_size, value_size = 32, 32, 32
#     norm_shape = [32]
#     train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
#
#     encoder = TransformerEncoder(
#         len(src_vocab), key_size, query_size, value_size, num_hiddens,
#         norm_shape, ffn_num_input, ffn_num_hiddens,num_heads,
#         num_layers, dropout)
#
#     decoder = TransformerDecoder(
#         len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
#         norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
#         num_layers, dropout)
#     net = d2l.EncoderDecoder(encoder, decoder)
#     d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
#
#
#     # 计算Bleu
#     engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
#     fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
#     for eng, fra in zip(engs, fras):
#         translation, dec_attention_weight_seq = d2l.predict_seq2seq(
#             net, eng, src_vocab, tgt_vocab, num_steps, device, True)
#         print(f'{eng} => {translation},',
#               f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
#
#     enc_attention_weights = torch.cat(net.encoder.attention_weights, 0).reshape((num_layers, num_heads,-1,num_steps))
#     print("transformer的注意力权重", enc_attention_weights.shape)
#
#     # d2l.plt.figure() # 创建一个新的绘图窗口
#     d2l.show_heatmaps(
#         enc_attention_weights.cpu(), xlabel='Key position',
#         ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],figsize=(7, 4))
#     d2l.plt.show()
