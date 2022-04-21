import os
import json
from argparse import ArgumentParser
import torch
import RNA
from source.data import *
from source.model import *

parser = ArgumentParser()
parser.add_argument("--config-path",help="path for configure json files.")
parser.add_argument("--model",help="model to use.")
parser.add_argument("--sequence",help="the query sequence(60-nt long only!).")
args = vars(parser.parse_args())
config_path = args["config_path"]
model_path = args["model"]
seq = args["sequence"]

model_config = json.load(open(os.path.join(config_path,"model.json"),"r"))
mid_channels = model_config["mid_channels"]
out_channels = model_config["out_channels"]
dropout = model_config["dropout"]
in_per_channel = model_config["in_per_channel"]

seq_conv = preNet_conv(mid_channels,out_channels,dropout)
struct_conv = preNet_conv(mid_channels,out_channels,dropout)
fcnn = preNet_linear(in_per_channel*out_channels,dropout)
model = preNet(seq_conv,struct_conv,fcnn)
model.load_state_dict(torch.load(model_path))
model = model.double()
model.eval()

struct = RNA.fold(seq)[0]
X_seq = sequence_encoder(seq).unsqueeze(0)
X_struct = struct_encoder(struct).unsqueeze(0)
score = model(X_seq,X_struct)
print(f"Estimated score of docking activity:{score.item()}")
