''' Define the Layers '''
import torch.nn as nn
import torch
import numpy as np
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward,SEQMultiHeadAttention,SEQ2MultiHeadAttention
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
cuda=torch.cuda.is_available()
__author__ = "Yu-Hsiang Huang"


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model_i, d_inner, n_head, d_k, d_v, n_position,BATCH_SIZE,dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.lstm = nn.LSTM(d_model_i+d_model_i, d_model_i,batch_first=True,bidirectional=True)
        self.pos_ffn = PositionwiseFeedForward(d_model_i+d_model_i, d_inner, dropout=dropout)
        self.d_model_i=d_model_i
        self.BATCH_SIZE=BATCH_SIZE
    def init_hidden(self,device):


        return (torch.zeros(2, self.BATCH_SIZE, self.d_model_i).to(device),torch.zeros(2, self.BATCH_SIZE, self.d_model_i).to(device))

    def forward(self, enc_input, slf_attn_mask=None):

        self.hidden=self.init_hidden(enc_input.device)
        textlen=enc_input.size(1)

        sentence=pack_padded_sequence(enc_input, slf_attn_mask, enforce_sorted=False,batch_first=True)

        
        lstm_out, self.hidden = self.lstm(sentence, self.hidden)
        lstm_out2,len_batch=pad_packed_sequence(lstm_out,batch_first=True,total_length=textlen)
        enc_output = self.pos_ffn(lstm_out2)

        return enc_output


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model_i,d_model_o, d_inner, n_head, d_k, d_v,videomaxlen,BATCH_SIZE,n_position,dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.lstm = nn.LSTM(d_model_o+d_model_o, d_model_o+d_model_o,batch_first=True)
        self.enc_attn =SEQ2MultiHeadAttention(n_head, d_model_o+d_model_o,d_model_i+d_model_i, d_k, d_v,videomaxlen,BATCH_SIZE, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model_o+d_model_o, d_inner, dropout=dropout)
        self.d_model_o=d_model_o
        self.BATCH_SIZE=BATCH_SIZE
    def init_hidden(self,device):

        return (torch.zeros(1, self.BATCH_SIZE, self.d_model_o+self.d_model_o).to(device),torch.zeros(1, self.BATCH_SIZE, self.d_model_o+self.d_model_o).to(device))

    def forward(
            self, dec_input, enc_output,index,
            slf_attn_mask=None, dec_enc_attn_mask=None):


        if index==0:
            self.hidden=self.init_hidden(dec_input.device)

        lstm_out, self.hidden = self.lstm(dec_input, self.hidden)

        dec_output, dec_enc_attn = self.enc_attn(lstm_out, enc_output, enc_output,index, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_enc_attn
