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
        #print(type(values[1:]),values[1:])
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
        #self.labellen=None
        tmpdata=[]
        self.raw_text=[]
        self.maxframelen=0
        file_list = sorted(glob.glob(data_folder+'/madevideo/*/'))
        for i in file_list[first:last]:
            #print(i)

            npz = np.load(i+'data.npz')
            textvec=torch.empty(0,len_word+1)
            with open(i+'story.txt') as f:
                raw_text=f.read()
                #print(raw_text)
                text=(raw_text).split(' ')
                for j in text:
                    #print(j)
                    textvec=torch.cat([textvec, (torch.cat([embeddings_dict[j],torch.Tensor([0])])).unsqueeze(0)])
            textvec=torch.cat([textvec, (torch.cat([torch.zeros(len_word),torch.Tensor([1])])).unsqueeze(0)])
            self.datanum+=1
            '''

            if(np.shape(npz['people'])[0]>maxlen):
                raise Exception("Length is too short id - {}, len - {}").format(self.dataset[item], video_len)
                continue
            '''

            frame=np.eye(MOVENUM)[(npz['people'])[:,:,0]]
            frame=frame.reshape([-1,2*9])
            if (self.maxframelen<np.shape(frame)[0]):
                self.maxframelen=np.shape(frame)[0]

            #print("1",np.shape(frame))
            frame=np.insert(frame,18,0,axis=1)
            #print("2",np.shape(frame))
            #frame=np.append(frame,np.append(np.ones((1,1)),np.zeros((1,2*12)),axis=1),axis=0)
            frame[-1][18]=1
            #print("3",np.shape(frame))
            frame2=torch.from_numpy(frame)
            time2=torch.from_numpy(npz['time'])
            time2=np.insert(time2,len(time2),0,axis=0)
            time2[-1][0]=len(frame)-1
            time2[-1][1]=len(frame)
            #print(frame2.type())


            #tmpdata.append((frame2,torch.from_numpy((npz['obj'][:,0:3].reshape([-1,6*3]))).float(),textvec,raw_text))
            tmpdata.append((frame2,textvec,raw_text,time2))


        #sorted_tmpdata=sorted(tmpdata, key=lambda x: (x[2].size()[0]),reverse=True)# in order to sort according to the text length.
        sorted_tmpdata=tmpdata
        for i in sorted_tmpdata:
            self.dataframe.append(i[0])

            self.label.append(i[1])
            self.labellen.append(len(i[1]))
            self.framelen.append(len(i[0]))
            self.raw_text.append(i[2])
            self.time.append(i[3])
        self.dataframe= pad_sequence(self.dataframe,batch_first=True,padding_value=float('inf'))
        self.time= pad_sequence(self.time,batch_first=True)

        self.labellen=torch.LongTensor(self.labellen)
        self.framelen=torch.LongTensor(self.framelen)

        #print(self.label)
        #print(self.labellen)
        self.label_t=pad_sequence(self.label,batch_first=True)
        self.framelen2=self.framelen.clone()
        self.path=torch.LongTensor(self.dataframe.size(0),self.dataframe.size(1))
        #print(self.dataframe.size())

        #print(self.labellen,self.label2.size(),self.labellen)



    def __len__(self):
        return self.datanum



    def __getitem__(self, idx):
        out_dataframe = self.dataframe[idx]

        out_label = self.label_t[idx]
        label_len=self.labellen[idx]
        frame_len=self.framelen[idx]
        frame_len2=self.framelen2[idx]
        time=self.time[idx]
        path=self.path[idx]

        if self.transform:
            out_datalabel = self.transform(out_data)
        return out_dataframe.float(),  out_label.float(),label_len,frame_len,time,frame_len2,path

        #sample = {'out_dataframe':out_dataframe, 'out_dataobj':out_dataobj, 'out_label':out_label,'label_len':label_len,'frame_len':frame_len}
        #return sample


    def getrawtext(self,idx):
        return self.raw_text[idx]
    def changelen(self,endframenum,path):
        self.framelen2=endframenum
        self.path=path

    def getmaxframe(self):
        return self.maxframelen
if __name__=="__main__":
    datan=Mydatasets("madevideo2")
