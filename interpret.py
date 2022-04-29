import json
from argparse import ArgumentParser
import torch
from source.data import *
from source.model import *
from source.grad_cam import *
from torch.utils.data import DataLoader
import pandas as pd

parser = ArgumentParser()
parser.add_argument("--config-path",help="path for configure json files.")
parser.add_argument("--model",help="model to use.")
parser.add_argument("--threshold",help="threshold score for searching the candidates")
args = vars(parser.parse_args())
config_path = args["config_path"]
model_path = args["model"]
threshold = eval(args["threshold"])

data_config = json.load(open(os.path.join(config_path,"data.json"),"r"))
model_config = json.load(open(os.path.join(config_path,"model.json"),"r"))

seq_fasta = data_config["seq_fasta"]
struct_fasta = data_config["struct_fasta"]
score_csv = data_config["score_csv"]

mid_channels = model_config["mid_channels"]
out_channels = model_config["out_channels"]
dropout = model_config["dropout"]
in_per_channel = model_config["in_per_channel"]

seqs,structs,scores = parse_and_split_samples(seq_fasta,struct_fasta,score_csv,split=False)
dataset = preNet_dataset(seqs,structs,scores)
dataloader = DataLoader(dataset)

seq_conv = preNet_conv(mid_channels,out_channels,dropout)
struct_conv = preNet_conv(mid_channels,out_channels,dropout)
fcnn = preNet_linear(in_per_channel*out_channels,dropout)
model = preNet(seq_conv,struct_conv,fcnn)
model.load_state_dict(torch.load(model_path))
model = model.double()

cam_seq = GradCAM(model,model.seq_net.conv2,info="seq")
cam_struct = GradCAM(model,model.struct_net.conv2,info="struct")

i = 0
s_ids,scores = [],[]
print(f"sample#\t\tlabel\t\tscore")
for X_seq,X_struct,y in dataloader:
    y = y.squeeze(0)
    score = cam_seq.forward(X_seq,X_struct,y,"results/grad_cam/",f"#{i+1}_seq.png")
    cam_struct.forward(X_seq,X_struct,y,"results/grad_cam/",f"#{i+1}_struct.png")
    print(f"{i+1}\t\t{y.item()}\t\t{score}")
    if y > 0.0 and score > threshold:
        s_ids.append(i+1)
        scores.append(score)
    i += 1
candidates = pd.DataFrame({"sample_id":s_ids,"score":scores})
candidates.to_csv("results/grad_cam/candidates.csv",sep="\t")
