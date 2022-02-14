import os
import json
import math
import glob

percent_init_state=0.5

import numpy as np
cur_time=0
videonums=10000

np.random.seed(0)

canuseperson=2

minimumlen_text=40

concat_set=['. ','. then ', ', and ']
def concat_text(i,text,timeofwords):
    if(len(text)==0):
        if(init_state[i]==False):
            init_state[i]=True
            if(len(init_text[i])>0):
                temp_before=len(text.split())
                subtext=init_text[i]+np.random.choice(['. ',', and '])
                timeofwords=np.append(timeofwords,np.ones((len(subtext.split()),2))*0,axis=0)
                timeofwords[temp_before:temp_before+len((init_text[i]).split()),1]=cur_time
                return subtext,timeofwords
            else:
                return '',timeofwords
        else:
            return '',timeofwords
    else:
        subtext=''

        if(init_state[i]==False):

            init_state[i]=True
            temp_before=len(text.split())
            if(len(init_text[i])>0):
                temp_before=len(text.split())
                subtext+='. '
                subtext=". "+init_text[i]+np.random.choice(['. ',', and '])
                timeofwords=np.append(timeofwords,np.ones((len(subtext.split()),2))*0,axis=0)
                timeofwords[temp_before+1:temp_before+1+len((init_text[i]).split()),1]=cur_time
                return subtext,timeofwords

        subtext+=np.random.choice(concat_set)
        timeofwords=np.append(timeofwords,np.ones((len(subtext.split()),2))*cur_time,axis=0)
        return subtext,timeofwords




def nextstate(oldframe,waiting,oldtext,timeofwords,text_begin_and_end_for_eachperson):
    canchangeperson=[]
    text=oldtext
    for i in range(canuseperson):

        if(oldframe[i][2]==0):
            tobechanged[i]=True
        if(tobechanged[i] and len(nextact[i])!=0):
            tmpframe[i][1]=0
            tmpframe[i][0]=nextact[i][0][0]
            tmpframe[i][2]=nextact[i][0][1]
            text_begin_and_end_for_eachperson[i][0]=len(text.split())-1
            text_begin_and_end_for_eachperson[i][1]=len(text.split())-1
            if(len(nextact[i][0][2])>0):

                appendtext,timeofwords=concat_text(i,text,timeofwords)
                text+=appendtext


                temp_before=len(text.split())
                text_begin_and_end_for_eachperson[i][0]=len(text.split())
                text+=person[i]+' '
                text+=nextact[i][0][2]
                timeofwords=np.append(timeofwords,np.ones((len(text.split())-temp_before,2))*cur_time,axis=0)
                text_begin_and_end_for_eachperson[i][1]=len(text.split())

            if(nextact[i][0][0]==5):

                text_begin_and_end_for_eachperson[i][1]=len(text.split())
                timeofwords=np.append(timeofwords,np.ones((2,2))*cur_time,axis=0)

            nextact[i].pop(0)
            tobechanged[i]=False



        elif(not(tobechanged[i]) or waiting):
            tmpframe[i][1]=oldframe[i][1]+1
            tmpframe[i][2]=oldframe[i][2]-1

        elif(not(waiting) and tobechanged[i] and len(nextact[i])==0):
            tmpframe[i][1]=oldframe[i][1]+1
            tmpframe[i][2]=oldframe[i][2]-1
            canchangeperson.append(i)

        else:
            print("something wrong!",waiting,tobechanged[i],len(nextact[i]),i)

    
    #state of the action(0),duration of the current action(1),waitig-time(2),

    changeperson=np.random.permutation(canchangeperson)

    for i in (changeperson):
        act[i]=[]
        if(waiting):
            break
        if(oldframe[i][0]==2):
            act[i].append([[0,min_state[0]+np.round(np.random.randint(0,var_state[0])),'began running ']])#begin running
            act[i].append([[1,min_state[1]+np.round(np.random.randint(0,var_state[1])),'began walking ']])#begin walking
            act[i].append([[6,min_state[0]+np.round(np.random.randint(0,var_state[0])),'began running while waving ']])#begin running and wave
            act[i].append([[7,min_state[1]+np.round(np.random.randint(0,var_state[1])),'began walking while waving ']])#begin walking and wave
            act[i].append([[8,2,'waved '],[2,min_state[2]+np.round(np.random.randint(0,var_state[2])),'']])
            act[i].append([[5,9,'sat down '],[3,min_state[3]+np.round(np.random.randint(0,var_state[3])),'']])

        elif(oldframe[i][0]==0 or oldframe[i][0]==6):
            act[i].append([[2,min_state[2]+np.round(np.random.randint(0,var_state[2])),'stopped ']])#stop
        elif(oldframe[i][0]==1 or oldframe[i][0]==7):
            act[i].append([[2,min_state[2]+np.round(np.random.randint(0,var_state[2])),'stopped ']])#stop
        elif(oldframe[i][0]==3):
            act[i].append([[4,9,'stood up '],[2,min_state[2]+np.round(np.random.randint(0,var_state[2])),'']])#stand up



        act_to_take=act[i][np.random.randint(len(act[i]))]
        if(len(act_to_take[0][2])>0):
            appendtext,timeofwords=concat_text(i,text,timeofwords)
            text+=appendtext
            temp_before=len(text.split())
            text_begin_and_end_for_eachperson[i][0]=len(text.split())

            text+=person[i]+' '


            text+=act_to_take[0][2]
            timeofwords=np.append(timeofwords,np.ones((len(text.split())-temp_before,2))*cur_time,axis=0)

            text_begin_and_end_for_eachperson[i][1]=len(text.split())




        nextact[i]=act_to_take[1:]

        tmpframe[i][0]=act_to_take[0][0]
        tmpframe[i][1]=0
        tmpframe[i][2]=act_to_take[0][1]

        tobechanged[i]=False


    return text,waiting,timeofwords
#state of the action,duration of the current action,waitig-time,
#move=[run0,walk1,standing2,sitting3,standfromsiton4,siton5,wave_while_run6,wave_while_walk7,wave8]









#below are minimum and variance of waiting time for each state(standing,sitting,walking, running)
min_sit=5
min_stand=4
min_walk=6
min_run=5
var_sit=12
var_stand=17
var_walk=22
var_run=14
min_state=[min_run,min_walk,min_stand,min_sit]
var_state=[var_run,var_walk,var_stand,var_sit]

person=["mike","jenny"]
text=''




img_outdir = './madevideo/'
os.makedirs(img_outdir, exist_ok=True)



people_frame=np.zeros((canuseperson,3),dtype=np.int32)#state of the action, duration of the current action,waitig-time,


for videonum in range(videonums):
    os.makedirs('{}/video{:07d}/'.format(img_outdir,videonum), exist_ok=True)
    frame=np.empty((0,people_frame.shape[0],people_frame.shape[1]),dtype=np.int32)#array to sustain info of frame for each time. to be put in video array later

    text=""
    tmpframe=people_frame.copy()
    init_text=["" for i in range(canuseperson)]
    text_begin_and_end_for_eachperson=[[0,0] for i in range(canuseperson)]
    timeofwords=np.empty((0,2))#indicate where each word is related to


    for i in range(canuseperson):
        if(tmpframe[i][0]!=3):
            a=np.random.choice(3)
            tmpframe[i][0]=a
            tmpframe[i][2]=min_state[a]+np.round(np.random.randint(0,var_state[a]))
            if(a==0):
                init_text[i]="{} was running around ".format(person[i])
            elif(a==1):
                init_text[i]="{} was walking around ".format(person[i])


    init_state=[False for j in range(canuseperson)]
    cur_time=0
    for i in (np.random.permutation(np.arange(canuseperson))):
        if(np.random.rand()>=percent_init_state):
            if(len(init_text[i])>0):
                temp_before=len(text.split())

                if(len(text)>0):
                    text+=np.random.choice(['. ',', and '])
                text_begin_and_end_for_eachperson[i][0]=len(text.split())

                text+=init_text[i]
                timeofwords=np.append(timeofwords,np.ones((len(text.split())-temp_before,2))*cur_time,axis=0)
                text_begin_and_end_for_eachperson[i][1]=len(text.split())


                init_state[i]=True







    waiting=False#if True, new action cannot be occurred.
    tobechanged=[False for i in range(canuseperson)]
    nextact=[[] for i in range(canuseperson)]
    act=[[] for i in range(canuseperson)]
    make_continue=True


    while(make_continue):
        for i in range(canuseperson):
            timeofwords[text_begin_and_end_for_eachperson[i][0]:text_begin_and_end_for_eachperson[i][1],1]+=1
        if(len(text.split(' '))>minimumlen_text and all(init_state[i]==True for i in range(canuseperson))):
            waiting=True
        frame=np.append(frame,[tmpframe],axis=0)
        cur_time+=1
        oldframe=tmpframe.copy()

        text,waiting,timeofwords=nextstate(oldframe,waiting,text,timeofwords,text_begin_and_end_for_eachperson)

        if(len(text.split(' '))>minimumlen_text and all(len(nextact[i])==0 for i in range(canuseperson)) and (all(tobechanged)) and all(init_state[i]==True for i in range(canuseperson))):
            make_continue=False

    temp_before=len(text.split())

    text+='.'
    timeofwords=np.append(timeofwords,np.ones((len(text.split())-temp_before,2))*cur_time-1,axis=0)
    print(text)
    path_txt='{}/video{:07d}/story.txt'.format(img_outdir,videonum)
    with open(path_txt, mode='w') as f:
        f.write(text)
    np.savez('{}/video{:07d}/data'.format(img_outdir,videonum), people=frame,time=timeofwords)
    npz = np.load('{}/video{:07d}/data.npz'.format(img_outdir,videonum))
