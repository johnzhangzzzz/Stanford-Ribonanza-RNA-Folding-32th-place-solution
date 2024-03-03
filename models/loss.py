import torch

def loss(pred,target):
    
    '''target['sn']=target['sn'].pow(pow_N_sn)
    target_sn=target['sn'].reshape(-1,1,pred.shape[-1])
    target_sn=target_sn.repeat(1,pred.shape[-2],1)
    target_sn = target_sn[target['mask'][:,:target_sn.shape[1]]]'''
    
    #pred_sn=pred*target_sn
    p = pred[target['mask'][:,:pred.shape[1]]]
    y = target['react'][target['mask']].clip(0,1)
    loss = (p-y).abs() ##*target_sn
    #loss = F.l1_loss(p, y, reduction='none')
    loss = loss[~torch.isnan(loss)].mean()
    return loss