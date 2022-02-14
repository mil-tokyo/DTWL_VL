''' D1efine the Transformer model '''
import torch
gpu_ids=[3]
BATCH_SIZE=100

import os
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer
import withtime_batch_dataload_inside as dataload
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import detect_anomaly
writer = SummaryWriter()
CLIP_MODEL=100
num_train=8000
eval_train=1000
DATA_NUM=10000
num_test=1000


training_data = dataload.Mydatasets("../2var_shortwave_text40_rec2",0,num_train)
eval_data = dataload.Mydatasets("../2var_shortwave_text40_rec2",num_train,eval_train+num_train)
test_data = dataload.Mydatasets("../2var_shortwave_text40_rec2",num_train+eval_train,DATA_NUM)
train_loader = torch.utils.data.DataLoader(dataset=training_data,
                         batch_size=BATCH_SIZE, shuffle=True)
eval_loader = torch.utils.data.DataLoader(dataset=eval_data,
                         batch_size=BATCH_SIZE, shuffle=True)
videomaxlen=(int)(max(training_data.maxframelen,test_data.maxframelen,eval_data.maxframelen)*1.2)

train_loader = torch.utils.data.DataLoader(dataset=training_data,
                         batch_size=BATCH_SIZE, shuffle=True)


frame, sentence,labellen,framelen,time= training_data[0]
n_position_sentence=sentence.size()[0]+5
n_position_frame=videomaxlen+5
n_position=max(n_position_frame,n_position_sentence)
filename='checkpoint.pth.tar'
D_model_s=51
D_model_f=19

DecayRate=0.985
END_THRE=0.5

P_end=0.01
Weight_end=4
P_content=2


epsilon=1.0e-10


torch.manual_seed(1)
cuda=torch.cuda.is_available()

if cuda:
    device = torch.device(f'cuda:{gpu_ids[0]}')
else:
    device = torch.device('cpu')


__author__ = "Yu-Hsiang Huang"

class IndexPositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position):
        super(IndexPositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))


    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x,index):
        return torch.cat((x ,((self.pos_table[:,index:index+1, :x.size(2)]).clone().detach()).repeat(x.size()[0],1,1)),2)

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))


    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return torch.cat((x ,(self.pos_table[:,:x.size(1), :x.size(2)].clone().detach()).repeat(x.size()[0],1,1)),2)


def get_pad_mask(seq, seq_len):
    pad=torch.zeros(seq.size()[0],seq.size()[1])
    for i in range(seq.size()[0]):
        for j in range(seq_len[i]):
            pad[i][j]=1
    if cuda:
        return pad.bool().unsqueeze(-2).detach().to(seq.device)
    else:
        return pad.bool().unsqueeze(-2).detach()


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()[0],seq.size()[1]
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    if cuda:
        return subsequent_mask.detach().to(seq.device)
    else:
        return subsequent_mask.detach()



class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,d_word_vec, n_layers, n_head, d_k, d_v,
            d_model_i, d_inner,dropout=0.1, n_position=n_position):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(d_model_i, n_position)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model_i, d_inner, n_head, d_k, d_v,n_position,BATCH_SIZE, dropout=dropout)
            for _ in range(n_layers)])


    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward

        enc_output = self.position_enc(src_seq)
        enc_output = self.dropout(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, slf_attn_mask=src_mask)

        enc_output=enc_output+torch.cat((src_seq ,(torch.zeros(src_seq.size())).to(src_seq.device)),2)
        if return_attns:
            return enc_output

        return enc_output


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,d_word_vec, n_layers, n_head, d_k, d_v,
            d_model_i,d_model_o,d_frame, d_inner, videomaxlen, n_position=200, dropout=0.1):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.videomaxlen=videomaxlen
        self.sig=nn.Sigmoid()

        self.trg_word_prj = nn.Linear(d_model_o+d_model_o, d_frame, bias=False)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model_i,d_model_o, d_inner, n_head, d_k, d_v,videomaxlen,BATCH_SIZE,n_position,dropout=dropout)
            for _ in range(n_layers)])

        self.position_enc = IndexPositionalEncoding(d_model_o, n_position)

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):


        # -- Forward


        for index in range(self.videomaxlen):
            residual=trg_seq[:,index:index+1]
            trg_seq2=self.position_enc(trg_seq[:,index:index+1],index)
            dec_output = self.dropout(trg_seq2)

            for dec_layer in self.layer_stack:
                dec_output, dec_enc_attn = dec_layer(
                    dec_output, enc_output,index, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)

            dec_output=self.trg_word_prj(dec_output)
            dec_output=dec_output+residual
            dec2=torch.zeros(dec_output.size())

            if cuda:
                dec2=dec2.to(dec_output.device)

            dec2[:,:,0:9]=torch.softmax(dec_output[:,:,0:9],2)
            dec2[:,:,9:18]=torch.softmax(dec_output[:,:,9:18],2)
            dec2[:,:,18:]=self.sig(dec_output[:,:,18:])


            trg_seq[:,index+1:index+2]=dec2

            if index==0:
                dec_enc_attn_list=dec_enc_attn

            else:
                dec_enc_attn_list = torch.cat((dec_enc_attn_list,dec_enc_attn),2)


        return trg_seq,  dec_enc_attn_list

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            d_word_vec=D_model_s, d_model_i=D_model_s,d_model_o=D_model_f, d_inner=2048,
            n_layers_enc=1,n_layers_dec=1, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=n_position,d_frame=D_model_f
            ):

        super().__init__()

        self.encoder = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model_i=d_model_i, d_inner=d_inner,
            n_layers=n_layers_enc, n_head=n_head, d_k=d_k, d_v=d_v,
             dropout=dropout)

        self.decoder = Decoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model_i=d_model_i,d_model_o=d_model_o, d_inner=d_inner,d_frame=d_frame,
            n_layers=n_layers_dec, n_head=n_head, d_k=d_k, d_v=d_v,
             dropout=dropout,videomaxlen=videomaxlen)


        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model_i == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'



    def forward(self, src_seq, trg_seq,labellen,framelen):


        src_mask = labellen
        src_mask2 = get_pad_mask(src_seq, labellen)
        trg_mask = get_subsequent_mask(trg_seq)

        enc_output= self.encoder(src_seq, src_mask)#attn_array.size()=(BATCH_SIZE,mxvideolen,textlen)
        dec_output, attn_array= self.decoder(trg_seq, trg_mask, enc_output, src_mask2)

        del src_mask
        del trg_mask


        return dec_output,attn_array

if __name__ == '__main__':
#with detect_anomaly():

    print("begin")

    model=Transformer()

    G_optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer=G_optimizer, gamma=DecayRate)

    G_evalloss=None
    epoch = 0
    loss_count=0
    if cuda:
        model.to(device)
        #model=torch.nn.DataParallel(model,gpu_ids)

    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        G_optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['random'])
        G_evalloss = checkpoint['evalloss'].to(device)
        scheduler.load_state_dict(checkpoint['scheduler'])
        loss_count=checkpoint['loss_count']
        for state in G_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))



    while(epoch<250):
        model.train()
        if loss_count>30:



            break
        G_loss=0
        iter=0
        print("epoch=",epoch)
        for frame,sentence,labellen,framelen,time in train_loader:
            iter+=1
            if cuda:
                frame,sentence,labellen,framelen=frame.float().to(device),sentence.float().to(device),labellen.to(device),framelen.to(device)
            else:
                frame,sentence=frame.float(),sentence.float()


            G_optimizer.zero_grad()

            input=torch.zeros(BATCH_SIZE,videomaxlen,D_model_f)


            if cuda:
                input=input.to(device)



            output,attn_array=model(sentence,input,labellen,framelen)
            fake_video=output[:,1:,:]


            framecomp_num=framelen.clone().int()
            G_difloss=0

            G_endloss=0

            ifend=fake_video[:,:,-1]


            for i in range(BATCH_SIZE):
                G_difloss+=-1*torch.mean(frame[i][:framecomp_num[i],0:9]*(torch.log(fake_video[i][:framecomp_num[i],0:9]*(1-9*epsilon)+epsilon)))-torch.mean(frame[i][:framecomp_num[i],9:18]*(torch.log(fake_video[i][:framecomp_num[i],9:18]*(1-9*epsilon)+epsilon)))

                G_endloss+=-1*torch.mean(Weight_end*(torch.log(ifend[i][framecomp_num[i]-1]*(1-2*epsilon)+epsilon))+torch.sum(torch.log((1-ifend[i][:framecomp_num[i]-1])*(1-2*epsilon)+epsilon)))




            writer.add_scalar('end', P_end*G_endloss, epoch*(num_train//BATCH_SIZE)+iter)
            writer.add_scalar('state', P_content*G_difloss, epoch*(num_train//BATCH_SIZE)+iter)
            G_loss=P_end*G_endloss+P_content*G_difloss
            G_loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(),CLIP_MODEL)
            G_optimizer.step()


        G_running_loss = G_loss.data


        print("G_running_loss=",G_running_loss)



        with torch.no_grad():
            model.eval()
            G_temploss=0
            G_difsum=0
            G_endsum=0

            for frame,sentence,labellen,framelen,time in eval_loader:
                if cuda:
                    frame,sentence,labellen,framelen=frame.float().to(device),sentence.float().to(device),labellen.to(device),framelen.to(device)
                else:
                    frame,sentence=frame.float(),sentence.float()

                G_optimizer.zero_grad()

                input=torch.zeros(BATCH_SIZE,videomaxlen,D_model_f)


                if cuda:
                    input=input.to(device)




                output, *_=model(sentence,input,labellen,framelen)
                fake_video=output[:,1:,:]


                framecomp_num=framelen.clone().int()
                G_difloss=0

                G_endloss=0
                ifend=fake_video[:,:,-1]
                for i in range(BATCH_SIZE):
                    G_difloss+=-1*torch.mean(frame[i][:framecomp_num[i],0:9]*(torch.log(fake_video[i][:framecomp_num[i],0:9]*(1-9*epsilon)+epsilon)))-torch.mean(frame[i][:framecomp_num[i],9:18]*(torch.log(fake_video[i][:framecomp_num[i],9:18]*(1-9*epsilon)+epsilon)))

                    G_endloss+=-1*torch.mean(Weight_end*(torch.log(ifend[i][framecomp_num[i]-1]*(1-2*epsilon)+epsilon))+torch.sum(torch.log((1-ifend[i][:framecomp_num[i]-1])*(1-2*epsilon)+epsilon)))

                G_difsum+=P_content*G_difloss.data
                G_endsum+=P_end*G_endloss.data
                G_temploss+=P_end*G_endloss+P_content*G_difloss
            writer.add_scalar('evalend',G_endsum, epoch)
            writer.add_scalar('evalstate', G_difsum, epoch)
            if (G_evalloss==None or G_temploss<G_evalloss):
                torch.save(model.state_dict(), "text20_res_model.pt")
                #torch.save(model.module.state_dict(), "text20_res_model.pt")
                print(G_evalloss,G_temploss)
                G_evalloss=G_temploss
                loss_count=0


            else:
                loss_count+=1





        epoch+=1

        scheduler.step()
        state = {'epoch': epoch, 'state_dict': model.state_dict(),'optimizer': G_optimizer.state_dict(), 'random': torch.get_rng_state(), 'evalloss': G_evalloss.to('cpu'),'scheduler': scheduler.state_dict(),'loss_count':loss_count,}
        torch.save(state, filename)



    writer.close()
    model.load_state_dict(torch.load("text20_res_model.pt"))





    with torch.no_grad():
        model.eval()
        for i in range(num_test//BATCH_SIZE):

            frame,sentence,labellen,framelen,time= test_data[(i)*BATCH_SIZE:(i+1)*BATCH_SIZE]
            if cuda:

                frame,sentence,labellen,framelen=frame.float().to(device),sentence.float().to(device),labellen.to(device),framelen.to(device)
            else:
                frame,sentence=frame.float(),sentence.float()



            input=torch.zeros(BATCH_SIZE,videomaxlen,D_model_f)

            if cuda:
                input=input.to(device)



            fake_video, attn_array=model(sentence,input,labellen,framelen)
            fake_video=fake_video[:,1:,:]
            ifend=fake_video[:,:,-1]

            for j in range(BATCH_SIZE):
                tl=torch.where(fake_video[j,:,18]>END_THRE)[0]

                if(len(tl)==0):
                    endframenum=(videomaxlen+2)
                else:
                    endframenum=(int(tl[0])+1)
                path_name='text20_generatadvideo/{:07d}/'.format((num_train+i)*BATCH_SIZE+j)
                os.makedirs(path_name, exist_ok=True)
                path_txt=path_name+'story.txt'

                with open(path_txt, mode='w') as f:
                    f.write(test_data.getrawtext((i)*BATCH_SIZE+j))
                np.savez(path_name+'data', people=fake_video[j].to('cpu').detach().numpy().copy(),gt=frame[j].to('cpu').detach().numpy().copy(),\
                gen_framelen=endframenum,gt_framelen=framelen[j].to('cpu').detach().numpy().copy(),attn_array=attn_array[j].to('cpu').detach().numpy().copy())
