import torch
epsilon=1.0e-10
global path
global match_pos2
P_content=2
WrongWeightPenal=3.2
import torch.nn as nn
class SDTWVL(nn.Module):
    def __init__(self):
        super(SDTWVL, self).__init__()

    def forward(self,gen,gt,endframenum,framelen,if_training,v_lambda=0.01,ifmatched=1.0,if_firstphase=False,P_end=0.2,requires_path=False,path=None):

        if (if_training and (not if_firstphase)):

            gt_rec=torch.gather(gt,1,(path.to(gt.device)).unsqueeze(-1).repeat(1,1,gt.size(2)))
            loss=P_content*(-1*torch.mean(gt_rec[:,:,0:-1]*torch.log(gen[:,:,0:-1]*(1-9*epsilon)+epsilon),-1)+torch.mean(gt_rec[:,:,0:-1]*torch.log(gt_rec[:,:,0:-1]*(1-9*epsilon)+epsilon),-1))*2+P_end*(-1*torch.mean(((gt_rec[:,:,-1:]*torch.log(gen[:,:,-1:]*(1-2*epsilon)+epsilon)))+((1-gt_rec[:,:,-1:])*torch.log((1-gen[:,:,-1:])*(1-2*epsilon)+epsilon))    ,-1))
            ans=torch.gather(torch.cumsum(loss,1),1,((torch.LongTensor(endframenum)-1).to(gt.device)).unsqueeze(1))
            return ans,torch.LongTensor(endframenum)-1,torch.zeros(gt.size(0))




        def dist_t(p1_b,p2_b):

            p1_mask=torch.where(p1_b==float('inf'),torch.Tensor([0]).to(p1_b.device),torch.Tensor([1]).to(p1_b.device))
            p2_mask=torch.where(p2_b==float('inf'),torch.Tensor([0]).to(p1_b.device),torch.Tensor([1]).to(p1_b.device))
            mask=p1_mask*p2_mask
            p1_b2, p2_b2 = torch.broadcast_tensors(p1_b, p2_b)
            p1_b3=torch.where(mask==0,mask,p1_b2)
            p2_b3=torch.where(mask==0,mask,p2_b2)
            assert len(p1_b3.size())==4
            p1=p1_b3[:,:,:,:-1]
            p2=p2_b3[:,:,:,:-1]
            p1e=p1_b3[:,:,:,-1:]
            p2e=p2_b3[:,:,:,-1:]

            return P_content*(-1*torch.mean(p2*torch.log(p1*(1-9*epsilon)+epsilon),-1)+torch.mean(p2*torch.log(p2*(1-9*epsilon)+epsilon),-1))*2+P_end*(-1*torch.mean(((p2e*torch.log(p1e*(1-2*epsilon)+epsilon)))+((1-p2e)*torch.log((1-p1e)*(1-2*epsilon)+epsilon))    ,-1)) -P_end*(-1*torch.mean(((p2e*torch.log(p1e*(1-2*epsilon)+epsilon)))+((1-p2e)*torch.log((1-p1e)*(1-2*epsilon)+epsilon))    ,-1)) .detach()

        for_preventing_lastdeletemask=((torch.eye(gt.size(1)+3).to(gt.device))[framelen]).unsqueeze(1).repeat(1,gen.size(1)+3,1)
        for_preventing_lastdeletemask2=((torch.eye(gt.size(1)+3).to(gt.device))[framelen-1]).unsqueeze(1).repeat(1,gen.size(1)+3,1)
        for_preventing_lastdelete=torch.where(for_preventing_lastdeletemask==1,torch.Tensor([1]).to(gt.device)*float('inf'),torch.Tensor([0]).to(gt.device))
        for_preventing_lastdelete2=torch.where(for_preventing_lastdeletemask2==1,torch.Tensor([1]).to(gt.device)*float('inf'),torch.Tensor([0]).to(gt.device))



        a=torch.zeros(len(gt),gen.size(1)+2,gt.size(1)+1).to(gt.device)
        del_ratio_map=torch.zeros(len(gt),gen.size(1)+1,gt.size(1)+1).to(gt.device)
        Path_save=torch.zeros(len(gt),gen.size(1)+1,gt.size(1)+1).to(gt.device)
        del_ratio_map[:,0,:]=torch.arange(gt.size(1)+1).to(gt.device).unsqueeze(0).repeat(len(gt),1)
        del_ratio_map[:,:,0]=torch.arange(gen.size(1)+1).to(gt.device).unsqueeze(0).repeat(len(gt),1)
        ans=0




        sub_gen=torch.cat((torch.ones(len(gt),1,gen.size(2)).to(gen.device)*float('inf'),gen),1)



        del_penalty=torch.cumsum(torch.triu((torch.ones(1,gt.size(1)+1,gt.size(1)+1)*v_lambda).to(gt.device),diagonal=1),2)
        del_penalty2=torch.cumsum(torch.triu((torch.ones(1,gt.size(1)+1,gt.size(1)+1)*v_lambda).to(gt.device),diagonal=2),2)
        ignore_tril=torch.tril(torch.ones(1,gt.size(1)+1,gt.size(1)+1).to(gt.device)*float("inf"),diagonal=-1)
        ignore_tril2=torch.tril(torch.ones(1,gt.size(1)+1,gt.size(1)+1).to(gt.device)*float("inf"),diagonal=0)


        del_gt_lossmap=del_penalty+ignore_tril


        torch_defaultnext=torch.arange(0,gt.size(1)+1).unsqueeze(0).repeat(gt.size(0),1).to(gen.device)

        a[:,1:,0]=((torch.ones(gen.size(1)+1)*1.0e+20).to(gen.device).unsqueeze(0).repeat(gen.size(0),1))
        a[:,0,1:]=((torch.ones(gt.size(1))*1.0e+20).to(gen.device).unsqueeze(0).repeat(gen.size(0),1))
        for_del_gen=torch.zeros(len(gt),1+gt.size(1))


        keep_for_del_gen=torch.zeros(len(gt),gen.size(1)+1,1+gt.size(1)).long().to(gen.device)
        keep_k=torch.zeros(len(gt),gen.size(1)+1,1+gt.size(1)).long().to(gen.device)


        for_del_gen=for_del_gen.long().to(gen.device)
        lastmatch_genindex_delgen=torch.zeros(1+gt.size(1)).unsqueeze(0).repeat(gt.size(0),1).to(gen.device).long()
        lastmatch_gtindex_delgen=torch.zeros(1+gt.size(1)).unsqueeze(0).repeat(gt.size(0),1).to(gen.device).long()
        lastmatch_genindex_delgt=torch.zeros(1+gt.size(1)).unsqueeze(0).repeat(gt.size(0),1).to(gen.device).long()
        lastmatch_gtindex_delgt=torch.zeros(1+gt.size(1)).unsqueeze(0).repeat(gt.size(0),1).to(gen.device).long()
        lastmatch_genindex_match=torch.zeros(1+gt.size(1)).unsqueeze(0).repeat(gt.size(0),1).to(gen.device).long()
        lastmatch_gtindex_match=torch.zeros(1+gt.size(1)).unsqueeze(0).repeat(gt.size(0),1).to(gen.device).long()
        del_ratio_map_delgen=torch.arange(gt.size(1)+1).to(gt.device).unsqueeze(0).repeat(len(gt),1)
        del_ratio_map_delgt=torch.arange(gt.size(1)+1).to(gt.device).unsqueeze(0).repeat(len(gt),1)
        del_ratio_map_match=torch.arange(gt.size(1)+1).to(gt.device).unsqueeze(0).repeat(len(gt),1)

        lastmatch_genindex=torch.zeros(1+gt.size(1)).unsqueeze(0).repeat(gt.size(0),1).to(gen.device).long()
        lastmatch_gtindex=torch.zeros(1+gt.size(1)).unsqueeze(0).repeat(gt.size(0),1).to(gen.device).long()
        map_dist_simple=torch.zeros(len(gt),gen.size(1)+3,gt.size(1)+3).to(gt.device)

        map_dist_simple[:,1:-2,1:-2]=dist_t(gen.unsqueeze(2),gt.unsqueeze(1))
        a_match_prev=a[:,0].clone()
        del_gen_prev=a[:,0].clone()
        del_gt_prev=a[:,0].clone()

        if if_firstphase:
            for i in range(1,len(gen[0])+1):

                addgrad_mask1=torch.where(((i-lastmatch_genindex_match)>(torch_defaultnext+1-lastmatch_gtindex_match)),torch.LongTensor([1]).to(gt.device),torch.LongTensor([0]).to(gt.device))
                addgrad_mask2=torch.where(((i-lastmatch_genindex_match)<(torch_defaultnext+1-lastmatch_gtindex_match)),torch.LongTensor([1]).to(gt.device),torch.LongTensor([0]).to(gt.device))

                addgrad_prev1=-map_dist_simple[:,i-1,0:-3]+(map_dist_simple[:,i-1,1:-2]+map_dist_simple[:,i,2:-1])*(((i-lastmatch_genindex_match)-(torch_defaultnext+1-lastmatch_gtindex_match))[:,:-1])
                addgrad_prev2=-map_dist_simple[:,i+1,2:-1]-(map_dist_simple[:,i+1,1:-2]+map_dist_simple[:,i,0:-3])*(((i-lastmatch_genindex_match)-(torch_defaultnext+1-lastmatch_gtindex_match))[:,:-1])

                addgrad=addgrad_mask1[:,:-1]*addgrad_prev1+addgrad_mask2[:,:-1]*addgrad_prev2

                del addgrad_mask1
                del addgrad_mask2

                match_tensor=a_match_prev[:,:-1]+map_dist_simple[:,i,1:-2]+( addgrad-addgrad.detach() )+(torch.min(map_dist_simple[:,i+1,2:-1],torch.min(map_dist_simple[:,i-1,0:-3],torch.min(torch.min(map_dist_simple[:,i,1:-2],map_dist_simple[:,i,0:-3]),map_dist_simple[:,i-1,1:-2])))).detach()
                del_tensor_prev=del_gen_prev[:,1:]+map_dist_simple[:,i,1:-2]+for_preventing_lastdelete[:,i,1:-2]

                del_gen_tensor=torch.cat((a[:,i,0:1].clone(),del_tensor_prev),1)
                a_match=torch.cat((a[:,i,0:1].clone(),match_tensor),1)
                lastmatch_genindex_match=(torch.ones(torch_defaultnext.size()).to(gt.device)*i).long()
                lastmatch_gtindex_match=torch_defaultnext
                lastmatch_gtindex_delgt=lastmatch_gtindex_match
                lastmatch_genindex_delgt=lastmatch_genindex_match
                del_ratio_map_match[:,1:]=del_ratio_map_match[:,:-1].clone()
                del_ratio_map_delgt=del_ratio_map_match
                del_gt_prev=a_match







                del_gt_source=del_gt_prev.unsqueeze(-1).repeat(1,1,gt.size(1)+1)+ignore_tril2+ torch.cumsum( (torch.triu((map_dist_simple[:,i:i+1,:-2].detach()+for_preventing_lastdelete[:,i:i+1,:-2]+for_preventing_lastdelete2[:,i:i+1,:-2]).repeat(1,gt.size(1)+1,1),diagonal=1)),2 )
                k=torch.argmin((del_gt_source),1)
                del_gt_tensor=torch.gather(del_gt_source,1,k.unsqueeze(1)).squeeze(1)
                lastmatch_gtindex_delgt=torch.gather(lastmatch_gtindex_delgt,1,k)
                lastmatch_genindex_delgt=torch.gather(lastmatch_genindex_delgt,1,k)




                a_match_3way_index=torch.argmin(torch.cat((del_gen_tensor.unsqueeze(0),del_gt_tensor.unsqueeze(0),a_match.unsqueeze(0)),0),0)
                a_match_3way=torch.gather(torch.cat((del_gen_tensor.unsqueeze(0),del_gt_tensor.unsqueeze(0),a_match.unsqueeze(0)),0),0,a_match_3way_index.unsqueeze(0)).squeeze(0)
                a_match_3waydel=torch.gather(torch.cat(((lastmatch_genindex_delgen-lastmatch_gtindex_delgen).unsqueeze(0),(lastmatch_genindex_delgt-lastmatch_gtindex_delgt).unsqueeze(0),(lastmatch_genindex_match-lastmatch_gtindex_match).unsqueeze(0)),0),0,a_match_3way_index.unsqueeze(0)).squeeze(0)

                a_match_prev=a_match_3way+v_lambda*torch.abs((i-torch_defaultnext)-a_match_3waydel)



                del_gen_prev=torch.where(del_gen_tensor<a_match,del_gen_tensor,a_match)


                lastmatch_genindex_match_prev=torch.gather(torch.cat(((lastmatch_genindex_delgen).unsqueeze(0),(lastmatch_genindex_delgt).unsqueeze(0),(lastmatch_genindex_match).unsqueeze(0)),0),0,a_match_3way_index.unsqueeze(0)).squeeze(0)
                lastmatch_gtindex_match_prev=torch.gather(torch.cat(((lastmatch_gtindex_delgen).unsqueeze(0),(lastmatch_gtindex_delgt).unsqueeze(0),(lastmatch_gtindex_match).unsqueeze(0)),0),0,a_match_3way_index.unsqueeze(0)).squeeze(0)
                lastmatch_gtindex_delgen=torch.where(del_gen_tensor<a_match,lastmatch_gtindex_delgen,lastmatch_gtindex_match)
                lastmatch_genindex_delgen=torch.where(del_gen_tensor<a_match,lastmatch_genindex_delgen,lastmatch_genindex_match)

                lastmatch_genindex_match=lastmatch_genindex_match_prev
                lastmatch_gtindex_match=lastmatch_gtindex_match_prev


                a[:,i]=a_match_prev
                del_ratio_map_delgen=del_ratio_map_delgen+1
                del_ratio_map_delgt=torch.gather(del_ratio_map_delgt,1,k)+(torch.arange(gt.size(1)+1).to(gt.device)-k)


                del_ratio_map_match_prev=torch.gather(torch.cat(((del_ratio_map_delgen).unsqueeze(0),(del_ratio_map_delgt).unsqueeze(0),(del_ratio_map_match).unsqueeze(0)),0),0,a_match_3way_index.unsqueeze(0)).squeeze(0)
                del_ratio_map_delgen=torch.where(del_gen_tensor<a_match,del_ratio_map_delgen,del_ratio_map_match)
                del_ratio_map_match=del_ratio_map_match_prev

                del_ratio_map[:,i]=del_ratio_map_match

        else:

            for i in range(1,len(gen[0])+1):

                match_tensor=a[:,i-1,:-1]+map_dist_simple[:,i,1:-2]+(torch.min(map_dist_simple[:,i+1,2:-1],torch.min(map_dist_simple[:,i-1,0:-3],torch.min(torch.min(map_dist_simple[:,i,1:-2],map_dist_simple[:,i,0:-3]),map_dist_simple[:,i-1,1:-2])))).detach()

                del_tensor_prev=a[:,i-1,1:]+(map_dist_simple[:,i,1:-2]+for_preventing_lastdelete[:,i,1:-2])+v_lambda

                min_temp=torch.min(del_tensor_prev,match_tensor)
                for_del_gen[:,1:]=torch.argmin(torch.cat(((del_tensor_prev-min_temp).unsqueeze(0), ((match_tensor-min_temp)).unsqueeze(0)),0),0 )
                keep_for_del_gen[:,i]=for_del_gen.clone()

                a_prev_cat= torch.gather(torch.cat(((del_tensor_prev).unsqueeze(0), match_tensor.unsqueeze(0)),0),0,for_del_gen[:,1:].clone().unsqueeze(0)).squeeze(0)
                a_before_delgt=torch.cat((a[:,i,0:1].clone(),a_prev_cat),1)

                del_ratio_map[:,i,1:]=(1-for_del_gen[:,1:])*(del_ratio_map[:,i-1,1:]+1)+(for_del_gen[:,1:])*del_ratio_map[:,i-1,:-1]
                lastmatch_gtindex=(1-for_del_gen)*lastmatch_gtindex+for_del_gen*torch_defaultnext
                lastmatch_genindex=(1-for_del_gen)*lastmatch_genindex+for_del_gen*i


                del_gt_source=a_before_delgt.unsqueeze(-1).repeat(1,1,gt.size(1)+1)+ignore_tril+ del_penalty+torch.cumsum( (torch.triu((map_dist_simple[:,i:i+1,:-2].detach()+for_preventing_lastdelete[:,i:i+1,:-2]+for_preventing_lastdelete2[:,i:i+1,:-2]).repeat(1,gt.size(1)+1,1),diagonal=1)),2 )# +ifmatched*(delgt_ifmatched-delgt_ifmatched.detach())#+shape_weight*(pair_loss2-pair_loss2.detach())
                k=torch.argmin(del_gt_source,1)
                a[:,i]=torch.gather(del_gt_source,1,k.unsqueeze(1)).squeeze(1)

                keep_k[:,i]=k.clone()

                lastmatch_gtindex=torch.gather(lastmatch_gtindex,1,k)
                lastmatch_genindex=torch.gather(lastmatch_genindex,1,k)

                del_ratio_map[:,i]=torch.gather(del_ratio_map[:,i],1,k)+(torch.arange(gt.size(1)+1).to(gt.device)-k)


        delnum=0
        allnum=0
        frameindex=framelen-1
        a2=a[:,1:-1,1:]

        k=torch.gather(a2,2,frameindex.unsqueeze(-1).unsqueeze(-1).repeat(1,a2.size(1),1)).squeeze(-1)
        endframenum2=(torch.LongTensor(endframenum)-1).to(k.device)
        endframenum2=endframenum2.to(k.device)
        goodendpos=torch.argmin( (k+WrongWeightPenal*torch.abs(torch.arange(1,k.size(1)+1).to(k.device).unsqueeze(0)-torch.LongTensor(endframenum).to(k.device).unsqueeze(1))) ,1)





        if (if_training and if_firstphase):
            ans=torch.gather(k,1,goodendpos.unsqueeze(-1)).squeeze(-1)
            allnum=framelen+goodendpos
            del_ratiomap2=torch.gather(del_ratio_map[:,1:,1:],2,frameindex.unsqueeze(-1).unsqueeze(-1).repeat(1,a2.size(1),1)).squeeze(-1)
            del_ratiomap3=torch.gather(del_ratiomap2,1,goodendpos.unsqueeze(-1)).squeeze(-1)
        else:
            endframenum2=(torch.LongTensor(endframenum)-1).to(k.device)
            endframenum2=endframenum2.to(k.device)
            ans=torch.gather(k,1,endframenum2.unsqueeze(-1)).squeeze(-1)
            allnum=framelen+endframenum2
            del_ratiomap2=torch.gather(del_ratio_map[:,1:,1:],2,frameindex.unsqueeze(-1).unsqueeze(-1).repeat(1,a2.size(1),1)).squeeze(-1)
            del_ratiomap3=torch.gather(del_ratiomap2,1,endframenum2.unsqueeze(-1)).squeeze(-1)




        del_ratio=del_ratiomap3*1.0/allnum
        if not if_firstphase:
            goodendpos=endframenum2

        if (if_firstphase and if_training):
            for i in range(ans.size(0)):
                ans[i]=ans[i]+P_end*-1*torch.sum(torch.log(gen[i,:,-1][(int)(goodendpos[i]+1):(int)(goodendpos[i]+2)]*(1-2*epsilon)+epsilon))




        if (not requires_path):
            return ans,goodendpos,del_ratio
        match_pos=((torch.eye(gt.size(1)+1).to(gt.device))[framelen]).unsqueeze(1).repeat(1,gen.size(1)+1,1)*((torch.eye(gen.size(1)+1).to(gt.device))[torch.LongTensor(endframenum).to(k.device)]).unsqueeze(2).repeat(1,1,gt.size(1)+1)

        match_pos=match_pos.long().to(gt.device)

        for i in range(len(gen[0]),0,-1):

            match_pos[:,i]=(torch.eye(gt.size(1)+1).long().to(gt.device))[torch.sum(keep_k[:,i]*match_pos[:,i].clone(),-1)]
            match_pos[:,i-1]+=match_pos[:,i].clone()*(1-keep_for_del_gen[:,i])
            match_pos[:,i-1,:-1]+=match_pos[:,i,1:].clone()*keep_for_del_gen[:,i,1:]
        match_pos2=match_pos*((torch.arange(gt.size(1)+1).to(gt.device)).unsqueeze(0).unsqueeze(1).repeat(gen.size(0),gen.size(1)+1,1))

        path=torch.where((torch.sum(match_pos2[:,1:,1:],2)-1)>=0,(torch.sum(match_pos2[:,1:,1:],2)-1),torch.LongTensor([8]).to(gt.device))

        return ans,goodendpos,del_ratio,path
