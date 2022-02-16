import glob
import torch
import numpy as np

from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence
MOVENUM=9
W=700
H=400
#make word dict from pretrained glove
embeddings_dict = {}
with open("../glove2/glove.6B.50d.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        vector2=torch.from_numpy(vector)
        embeddings_dict[word] = vector2

len_word=vector2.size()[0]



class Mydatasets(torch.utils.data.Dataset):
    def __init__(self,data_folder,first,last, transform = None):
        self.transform = transform
        self.dataframe=[]

        self.label=[]
        self.labellen=[]
        self.framelen=[]
        self.time=[]
        self.datanum=0
        tmpdata=[]
        self.raw_text=[]
        self.maxframelen=0
        file_list = sorted(glob.glob(data_folder+'/madevideo/*/'))
        for i in file_list[first:last]:


            npz = np.load(i+'data.npz')
            textvec=torch.empty(0,len_word+1)
            with open(i+'story.txt') as f:
                raw_text=f.read()
                text=(raw_text).split(' ')
                for j in text:
                    textvec=torch.cat([textvec, (torch.cat([embeddings_dict[j],torch.Tensor([0])])).unsqueeze(0)])
            textvec=torch.cat([textvec, (torch.cat([torch.zeros(len_word),torch.Tensor([1])])).unsqueeze(0)])
            self.datanum+=1


            frame=np.eye(MOVENUM)[(npz['people'])[:,:,0]]
            frame=frame.reshape([-1,2*9])
            if (self.maxframelen<np.shape(frame)[0]):
                self.maxframelen=np.shape(frame)[0]

            frame=np.insert(frame,18,0,axis=1)
            frame[-1][18]=1
            frame2=torch.from_numpy(frame)
            time2=torch.from_numpy(npz['time'])
            time2=np.insert(time2,len(time2),0,axis=0)
            time2[-1][0]=len(frame)-1
            time2[-1][1]=len(frame)
            tmpdata.append((frame2,textvec,raw_text,time2))


        for i in tmpdata:
            self.dataframe.append(i[0])

            self.label.append(i[1])
            self.labellen.append(len(i[1]))
            self.framelen.append(len(i[0]))
            self.raw_text.append(i[2])
            self.time.append(i[3])
        self.dataframe= pad_sequence(self.dataframe,batch_first=True)
        self.time= pad_sequence(self.time,batch_first=True)

        self.labellen=torch.LongTensor(self.labellen)
        self.framelen=torch.LongTensor(self.framelen)

        self.label_t=pad_sequence(self.label,batch_first=True)


    def __len__(self):
        return self.datanum



    def __getitem__(self, idx):
        out_dataframe = self.dataframe[idx]

        out_label = self.label_t[idx]
        label_len=self.labellen[idx]
        frame_len=self.framelen[idx]
        time=self.time[idx]

        if self.transform:
            out_datalabel = self.transform(out_data)
        return out_dataframe.float(),  out_label.float(),label_len,frame_len,time



    def getrawtext(self,idx):
        return self.raw_text[idx]

    def getmaxframe(self):
        return self.maxframelen
if __name__=="__main__":
    datan=Mydatasets("madevideo2")
