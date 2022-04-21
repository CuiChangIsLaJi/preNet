import torch
from sklearn.metrics import *

class Trainer:
    def __init__(self,model,loss_fn,optimizer,scheduler,device=None):
        self.model = model
        self.loss = loss_fn
        self.optim = optimizer
        self.scheduler = scheduler
        self.device = device
    def train_loop(self,dataloader):
        self.model.train()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        avg_loss = 0.
        for batch,(X1,X2,y) in enumerate(dataloader):
            if self.device != None:
                X1,X2,y = X1.to(self.device),X2.to(self.device),y.to(self.device)

            pred = self.model(X1,X2)
            loss = self.loss(pred,y)
            avg_loss += loss.item() / num_batches

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        self.scheduler.step()
        return avg_loss
    def evaluate(self,dataloader,mode="eval"):
        self.model.eval()
        size = len(dataloader.dataset)
        avg_loss = 0.
        y_true,y_pred,y_score = [],[],[]
        with torch.no_grad():
            for X1,X2,y in dataloader:
                if self.device != None:
                    X1,X2,y = X1.to(self.device),X2.to(self.device),y.to(self.device)

                pred = self.model(X1,X2)
                avg_loss += self.loss(pred,y).item() / size

                y_true.append(y.item())
                y_score.append(pred.item())
                y_pred.append(1 if pred>0.5 else 0)

        if mode=="eval":
            f1 = f1_score(y_true,y_pred)
            mcc = matthews_corrcoef(y_true,y_pred)
            auc = roc_auc_score(y_true,y_score)
            fpr,tpr,_ = roc_curve(y_true,y_score)
            return f1,mcc,auc,(fpr,tpr)
        elif mode=="train":
            acc = accuracy_score(y_true,y_pred)
            return avg_loss,acc
    def save_model(self,path):
        torch.save(self.model.state_dict(),path + "/preNet.pth")
    def load_model(self,filename):
        model = torch.load(filename)
        return model
