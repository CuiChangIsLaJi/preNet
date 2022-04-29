import json
import os
from argparse import ArgumentParser
from source.data import *
from source.model import *
from source.train import *
from sklearn.metrics import *
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser()
parser.add_argument("--config-path",help="path for configure json files.")
parser.add_argument("--model-path",help="path for dumping the trained model.")
args = vars(parser.parse_args())
config_path = args["config_path"]
model_path = args["model_path"]

data_config = json.load(open(os.path.join(config_path,"data.json"),"r"))
model_config = json.load(open(os.path.join(config_path,"model.json"),"r"))
train_config = json.load(open(os.path.join(config_path,"train.json"),"r"))

seq_fasta = data_config["seq_fasta"]
struct_fasta = data_config["struct_fasta"]
score_csv = data_config["score_csv"]
batch_size = data_config["batch_size"]
test_size = data_config["test_size"]
seqs_train,seqs_test,structs_train,structs_test,scores_train,scores_test = parse_and_split_samples(seq_fasta,struct_fasta,score_csv,split=True,test_size=test_size)
training_set = preNet_dataset(seqs_train,structs_train,scores_train)
test_set = preNet_dataset(seqs_test,structs_test,scores_test)
training_dataloader = DataLoader(training_set,batch_size=batch_size)
test_dataloader = DataLoader(test_set)

mid_channels = model_config["mid_channels"]
out_channels = model_config["out_channels"]
dropout = model_config["dropout"]
in_per_channel = model_config["in_per_channel"]
seq_conv = preNet_conv(mid_channels,out_channels,dropout)
struct_conv = preNet_conv(mid_channels,out_channels,dropout)
fcnn = preNet_linear(in_per_channel*out_channels,dropout)
model = preNet(seq_conv,struct_conv,fcnn)
model = model.double()
model = model.kaiming_init()

init_lr = train_config["init_lr"]
num_epochs = train_config["num_epochs"]
step_size = train_config["step_size"]
gamma = train_config["gamma"]
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=init_lr)
scheduler = StepLR(optimizer,step_size,gamma)
trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler
        )
print("Start training a preNet model.")
print("Epoch #\t\ttrain loss\t\ttest loss\t\ttest accuracy")
for i in range(num_epochs):
    loss_train = trainer.train_loop(training_dataloader)
    loss_test,acc_test = trainer.evaluate(test_dataloader,mode="train")
    print(f"{i+1}\t\t{loss_train:>7f}\t\t{loss_test:>7f}\t\t{acc_test:>7f}")
trainer.save_model(model_path)

print("End training a preNet model.")
print("Start evaluating the preNet model.")
training_dataloader = DataLoader(training_set)
f1_train,mcc_train,auc_train,(fpr_train,tpr_train) = trainer.evaluate(training_dataloader)
f1_test,mcc_test,auc_test,(fpr_test,tpr_test) = trainer.evaluate(test_dataloader)
print("Metrics\t\ttrain\t\ttest")
print(f"F1\t\t{f1_train:>3f}\t{f1_test:>3f}")
print(f"MCC\t\t{mcc_train:>3f}\t{mcc_test:>3f}")
print(f"AUC\t\t{auc_train:>3f}\t{auc_test:>3f}")
pd.DataFrame({"fpr":fpr_train,"tpr":tpr_train}).to_csv("results/roc_train.csv",sep="\t",header=False,index=False)
pd.DataFrame({"fpr":fpr_test,"tpr":tpr_test}).to_csv("results/roc_test.csv",sep="\t",header=False,index=False)
print("End evaluating the preNet model.")
