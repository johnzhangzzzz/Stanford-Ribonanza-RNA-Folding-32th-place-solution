from fastai.vision.all import *

class MAE(Metric):
    '''accumulate时滤除mask值'''
    def __init__(self): 
        self.reset()
        
    def reset(self): 
        self.x,self.y = [],[]
        
    def accumulate(self, learn):
        x = learn.pred[learn.y['mask'][:,:learn.pred.shape[1]]]
        y = learn.y['react'][learn.y['mask']].clip(0,1)
        self.x.append(x)
        self.y.append(y)

    @property
    def value(self):
        x,y = torch.cat(self.x,0),torch.cat(self.y,0)
        loss = F.l1_loss(x, y, reduction='none')
        loss = loss[~torch.isnan(loss)].mean()
        return loss