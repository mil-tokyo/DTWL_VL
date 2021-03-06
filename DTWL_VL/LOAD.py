''' Define the Transformer model '''
import torch
gpu_ids=[1]
BATCH_SIZE=50

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
thre_attn=0.1

training_data = dataload.Mydatasets("../simple_time_textmin_rec2",0,num_train)
eval_data = dataload.Mydatasets("../simple_time_textmin_rec2",num_train,eval_train+num_train)
test_data = dataload.Mydatasets("../simple_time_textmin_rec2",num_train+eval_train,DATA_NUM)
train_loader = torch.utils.data.DataLoader(dataset=training_data,
                         batch_size=BATCH_SIZE, shuffle=True)
eval_loader = torch.utils.data.DataLoader(dataset=eval_data,
                         batch_size=BATCH_SIZE, shuffle=True)
videomaxlen=(int)(max(training_data.maxframelen,test_data.maxframelen,eval_data.maxframelen)*1.2)

train_loader = torch.utils.data.DataLoader(dataset=training_data,
                         batch_size=BATCH_SIZE, shuffle=True)
'''
test_loader = torch.utils.data.DataLoader(dataset=test_data[num_train:],
                         batch_size=BATCH_SIZE, shuffle=False)
'''

MAXLEN=(int)(videomaxlen*1.5)

frame, sentence,labellen,framelen,time= training_data[0]
n_position_sentence=sentence.size()[0]+5
textlen=sentence.size()[0]
n_position_frame=videomaxlen+5
n_position=max(n_position_frame,n_position_sentence)
filename='checkpoint.pth.tar'
D_model_s=51
D_model_f=19

DecayRate=0.985
END_THRE=0.5

P_DIFFRAME=0.01
P_DIFFRAME2=4
P_DIFFRAME3=2
P_DIFFRAME4=1
P_DIFFRAME_attn=0
BISHO=1.0e-10
W=700
H=400

torch.manual_seed(1)
cuda=torch.cuda.is_available()

if cuda:
    device = torch.device(f'cuda:{gpu_ids[0]}')
else:
    device = torch.device('cpu')

#cuda=False


__author__ = "Yu-Hsiang Huang"

class IndexPositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position):
        super(IndexPositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
        print(n_position,"line17")

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
        #print("x.size=",x.size(),self.pos_table[:,:x.size(1), :x.size(2)].size())
        #return x + self.pos_table[:,index:index+1, :x.size(2)].clone().detach()
        return torch.cat((x ,((self.pos_table[:,index:index+1, :x.size(2)]).clone().detach()).repeat(x.size()[0],1,1)),2)
        #return x + self.pos_table[:,:x.size(1), :x.size(2)].clone().detach()

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
        print(n_position,"line44")

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
        #print("x.size=",x.size(),self.pos_table[:,:x.size(1), :x.size(2)].size())

        return torch.cat((x ,(self.pos_table[:,:x.size(1), :x.size(2)].clone().detach()).repeat(x.size()[0],1,1)),2)


def get_pad_mask(seq, seq_len):
    pad=torch.zeros(seq.size()[0],seq.size()[1])
    for i in range(seq.size()[0]):
        for j in range(seq_len[i]):
            pad[i][j]=1
    if cuda:
        #return pad.bool().unsqueeze(-2).detach().to(device)
        return pad.bool().unsqueeze(-2).detach().to(seq.device)
    else:
        return pad.bool().unsqueeze(-2).detach()
    #return (seq != pad_idx).unsqueeze(-2)


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

        #self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)

        self.dropout = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(d_model_i, n_position)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model_i, d_inner, n_head, d_k, d_v,n_position, dropout=dropout)
            for _ in range(n_layers)])
        #self.layer_norm = nn.LayerNorm(d_model_i, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward

        enc_output = self.position_enc(src_seq)
        enc_output = self.dropout(enc_output)
        #residual=src_seq.clone()
        #enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            #enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        #print("enc_output",enc_output.size())
        enc_output=enc_output+torch.cat((src_seq ,(torch.zeros(src_seq.size())).to(src_seq.device)),2)
        if return_attns:
            return enc_output , enc_slf_attn.detach()
        return enc_output, enc_slf_attn.detach()


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,d_word_vec, n_layers, n_head, d_k, d_v,
            d_model_i,d_model_o,d_frame, d_inner, videomaxlen, n_position=200, dropout=0.1):

        super().__init__()

        #self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)

        self.dropout = nn.Dropout(p=dropout)
        self.videomaxlen=videomaxlen
        self.sig=nn.Sigmoid()

        self.trg_word_prj = nn.Linear(d_model_o+d_model_o, d_frame, bias=False)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model_i,d_model_o, d_inner, n_head, d_k, d_v,videomaxlen,BATCH_SIZE,n_position,dropout=dropout)
            for _ in range(n_layers)])

        self.position_enc = IndexPositionalEncoding(d_model_o, n_position)
        #self.layer_norm = nn.LayerNorm(d_frame, eps=1e-6)

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        #dec_slf_attn_list = []
        #dec_enc_attn_list=torch.Tensor((BATCH_SIZE,0,textlen))#[], []

        # -- Forward

        #dec_output = self.layer_norm(dec_output)
        for index in range(self.videomaxlen):
            residual=trg_seq[:,index:index+1]
            trg_seq2=self.position_enc(trg_seq[:,index:index+1],index)
            #print(index)
            dec_output = self.dropout(trg_seq2)

            #print("Model182,dec_output",dec_output.size())
            for dec_layer in self.layer_stack:
                dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                    dec_output, enc_output,index, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)

            dec_output=self.trg_word_prj(dec_output)
            dec_output=dec_output+residual
            dec2=torch.zeros(dec_output.size())
            #print("dec2.size()",dec2.size())
            if cuda:
                dec2=dec2.to(dec_output.device)
            #print("orig",torch.isnan(dec_output).any() or torch.isnan(dec_output).any())

            dec2[:,:,0:9]=torch.softmax(dec_output[:,:,0:9],2)
            dec2[:,:,9:18]=torch.softmax(dec_output[:,:,9:18],2)
            dec2[:,:,18:]=self.sig(dec_output[:,:,18:])

            #print("2",torch.isnan(dec2).any() or torch.isnan(dec2).any())

            #print("Modwl190attn_array,",dec_enc_attn.size())



            trg_seq[:,index+1:index+2]=dec2
            #dec_enc_attn,_=torch.max(dec_enc_attn,axis=1)

            #dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            if index==0:
                dec_enc_attn_list=dec_enc_attn
                dec_slf_attn_list=dec_slf_attn.detach()
            else:
                dec_enc_attn_list = torch.cat((dec_enc_attn_list,dec_enc_attn),2)
                dec_slf_attn_list = torch.cat((dec_slf_attn_list,dec_slf_attn.detach()),2)
        #print("dec_output",dec_output.size())
        return trg_seq,  dec_enc_attn_list,dec_slf_attn_list
        #if return_attns:
            #return trg_seq, dec_slf_attn_list, dec_enc_attn_list
        #return trg_seq,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            d_word_vec=D_model_s, d_model_i=D_model_s,d_model_o=D_model_f, d_inner=2048,
            n_layers_enc=1,n_layers_dec=1, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=n_position,d_frame=D_model_f
            ):

        super().__init__()

        #self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

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

        #self.trg_word_prj = nn.Linear(d_model_o+d_model_o, d_frame, bias=False)



        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model_i == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        #self.x_logit_scale = 1.
        '''
        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        '''

        '''
        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight
        '''


    def forward(self, src_seq, trg_seq,labellen,framelen):


        src_mask = get_pad_mask(src_seq, labellen)
        trg_mask = get_subsequent_mask(trg_seq)
        #trg_mask = get_pad_mask(trg_seq, framelen) & get_subsequent_mask(trg_seq)
        #print("Model189,mask",src_mask.size(),trg_mask.size())

        enc_output, enc_slf_attn_list= self.encoder(src_seq, src_mask)#attn_array.size()=(BATCH_SIZE,mxvideolen,textlen)
        dec_output, attn_array,dec_slf_attn_list = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        #seq_logit = self.trg_word_prj(dec_output) * self.x_logit_scale
        #print("attn_array,",attn_array.size())

        del src_mask
        del trg_mask


        return dec_output,attn_array,enc_slf_attn_list,dec_slf_attn_list

if __name__ == '__main__':
#with detect_anomaly():

    print("begin")

    model=Transformer()

    G_optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.5, 0.999))

    G_evalloss=None
    epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        G_optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['random'])
        G_evalloss = checkpoint['evalloss']
        for state in G_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))


    if cuda:
        model.to(device)
        #model=torch.nn.DataParallel(model,gpu_ids)
        #model.to(device)
        #model.to(device)
    while(epoch<250):
        model.train()
        G_loss=0
        iter=0
        print("epoch=",epoch)
        #count=0
        #for data in train_loader:
        for frame,sentence,labellen,framelen,time in train_loader:
            iter+=1
            if cuda:
                frame,sentence,labellen,framelen=frame.float().to(device),sentence.float().to(device),labellen.to(device),framelen.to(device)
            else:
                frame,sentence=frame.float(),sentence.float()


            #frame=frame[:,:,1:].float()

            G_optimizer.zero_grad()

            input=torch.zeros(BATCH_SIZE,videomaxlen,D_model_f)
            '''
            if cuda:
                input=input.to(device)
                fake_video=fake_video.to(device)
            '''


            if cuda:
                input=input.to(device)



            output,attn_array,enc_slf_attn_list,dec_slf_attn_list=model(sentence,input,labellen,framelen)
            #print(attn_array.size())

            fake_video=output[:,1:,:]


            framecomp_num=framelen.clone().int()
            G_difloss=0

            G_endloss=0
            G_attnloss=0
            ifend=fake_video[:,:,-1]


            #for i in range(BATCH_SIZE):
            #G_loss=G_loss+P_DIFFRAME*(torch.sum((frame[:framecomp_num,1:]-fake_video[:framecomp_num])*(frame[:framecomp_num,1:]-fake_video[:framecomp_num])))
            for i in range(BATCH_SIZE):
                G_difloss+=-1*torch.mean(frame[i][:framecomp_num[i],0:9]*(torch.log(fake_video[i][:framecomp_num[i],0:9]*(1-9*BISHO)+BISHO)))-torch.mean(frame[i][:framecomp_num[i],9:18]*(torch.log(fake_video[i][:framecomp_num[i],9:18]*(1-9*BISHO)+BISHO)))

                G_endloss+=-1*torch.mean(P_DIFFRAME2*(torch.log(ifend[i][framecomp_num[i]-1]*(1-2*BISHO)+BISHO))+torch.sum(torch.log((1-ifend[i][:framecomp_num[i]-1])*(1-2*BISHO)+BISHO)))

                for word in range(time[i].size()[0]):
                    wordnum_start=max((int)(time[i][word][0]),0)
                    wordnum_end=max((int)(time[i][word][1]),0)
                    if wordnum_start!=wordnum_end:
                        G_attnloss+=torch.mean((-1*torch.sum(attn_array[i,:,wordnum_start:wordnum_end,word],1)+torch.sum(attn_array[i,:,0:framecomp_num[i],word],1)))




            writer.add_scalar('end', P_DIFFRAME*G_endloss, epoch*(num_train//BATCH_SIZE)+iter)
            writer.add_scalar('attn', P_DIFFRAME_attn*G_attnloss, epoch*(num_train//BATCH_SIZE)+iter)

            writer.add_scalar('state', P_DIFFRAME3*G_difloss, epoch*(num_train//BATCH_SIZE)+iter)
            G_loss=P_DIFFRAME*G_endloss+P_DIFFRAME3*G_difloss+P_DIFFRAME_attn*G_attnloss
            G_loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(),CLIP_MODEL)
            G_optimizer.step()
            #print("optimizer step")

            #for dec_layer in model.module.decoder.layer_stack:
                #del dec_layer.slf_attn.k_stock
                #del dec_layer.slf_attn.v_stock
                #del dec_layer.enc_attn.k_stock
                #del dec_layer.enc_attn.v_stock

            del frame
            del sentence
            del framelen

            del labellen
            del fake_video
            del ifend
            del input

            del G_difloss

            del G_endloss
            torch.cuda.empty_cache()
            #print("del")
        G_running_loss = G_loss.data
        del G_loss

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
                '''
                if cuda:
                    input=input.to(device)
                    fake_video=fake_video.to(device)
                '''

                if cuda:
                    input=input.to(device)




                output, *_=model(sentence,input,labellen,framelen)
                fake_video=output[:,1:,:]


                framecomp_num=framelen.clone().int()
                G_difloss=0

                G_endloss=0
                ifend=fake_video[:,:,-1]
                for i in range(BATCH_SIZE):
                    G_difloss+=-1*torch.mean(frame[i][:framecomp_num[i],0:9]*(torch.log(fake_video[i][:framecomp_num[i],0:9]*(1-9*BISHO)+BISHO)))-torch.mean(frame[i][:framecomp_num[i],9:18]*(torch.log(fake_video[i][:framecomp_num[i],9:18]*(1-9*BISHO)+BISHO)))

                    G_endloss+=-1*torch.mean(P_DIFFRAME2*(torch.log(ifend[i][framecomp_num[i]-1]*(1-2*BISHO)+BISHO))+torch.sum(torch.log((1-ifend[i][:framecomp_num[i]-1])*(1-2*BISHO)+BISHO)))

                G_difsum+=P_DIFFRAME3*G_difloss.data
                G_endsum+=P_DIFFRAME*G_endloss.data
                G_temploss+=P_DIFFRAME*G_endloss+P_DIFFRAME3*G_difloss
            writer.add_scalar('evalend',G_endsum, epoch)
            writer.add_scalar('evalstate', G_difsum, epoch)
            if (G_evalloss==None or G_temploss<G_evalloss):
                torch.save(model.state_dict(), "text20_res_model.pt")
                print(G_evalloss,G_temploss)
                G_evalloss=G_temploss




                del frame
                del sentence
                del framelen

                del labellen
                del fake_video
                del ifend
                del input
                del G_temploss
                del G_difloss

                del G_endloss
                torch.cuda.empty_cache()

        epoch+=1

        #D_lr_scheduler.step()
        #state = {'epoch': epoch, 'state_dict': model.module.state_dict(),'optimizer': G_optimizer.state_dict(), 'random': torch.get_rng_state(), }
        state = {'epoch': epoch, 'state_dict': model.state_dict(),'optimizer': G_optimizer.state_dict(), 'random': torch.get_rng_state(), 'evalloss': G_evalloss,}
        torch.save(state, filename)

        #D_lr_scheduler.step()

    writer.close()
    #torch.save(model.module.state_dict(), "text20_res_model.pt")
    #torch.save(model.state_dict(), "text20_res_model.pt")
    model.load_state_dict(torch.load("text20_res_model.pt"))





    with torch.no_grad():
        model.eval()
        #D.eval()
        #for data in train_loader:
        for i in range(num_test//BATCH_SIZE):

            frame,sentence,labellen,framelen,time= test_data[(i)*BATCH_SIZE:(i+1)*BATCH_SIZE]
            if cuda:

                frame,sentence,labellen,framelen=frame.float().to(device),sentence.float().to(device),labellen.to(device),framelen.to(device)
            else:
                frame,sentence=frame.float(),sentence.float()



            input=torch.zeros(BATCH_SIZE,videomaxlen,D_model_f)
            image_t=torch.Tensor([W,H]).detach()
            if cuda:
                input=input.to(device)
                #fake_video=fake_video.to(device)
                image_t=image_t.to(device)



            fake_video, attn_array,enc_slf_attn_list,dec_slf_attn_list=model(sentence,input,labellen,framelen)
            fake_video=fake_video[:,1:,:]
            ifend=fake_video[:,:,-1]
            fake_video[:,:,0:9]=torch.eye(9)[torch.argmax(fake_video[:,:,0:9],2)]

            fake_video[:,:,9:18]=torch.eye(9)[torch.argmax(fake_video[:,:,9:18],2)]

            #fake_video[:,:,26:]=self.sig(fake_video[:,:,26:])

            for j in range(BATCH_SIZE):
                tl=torch.where(fake_video[j,:,18]>END_THRE)[0]
                #print("tl=",tl)

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
                gen_framelen=endframenum,gt_framelen=framelen[j].to('cpu').detach().numpy().copy(),attn_array=attn_array[j].to('cpu').detach().numpy().copy(),enc_slf_attn_list=enc_slf_attn_list[j].to('cpu').detach().numpy().copy(),dec_slf_attn_list=dec_slf_attn_list[j].to('cpu').detach().numpy().copy())
