# preNet
  This is a part of my graduation project, where you can train a Convolution Network named preNet to predict and understand the mechanism of pre-let-7/LIN28A axis. Welcome to download, run and evaluate my work !
# Structure of the repository
  train_and_test.py  
  interpret.py  
  predict.py  
  source  
  |----data.py  
  |----model.py  
  |----train.py  
  |----grad_cam.py  
  data  
  |----average_scores.csv  
  |----sequences.fa  
  |----structures.fa  
  model  
  |----preNet.pth  
  config  
  |----data.json  
  |----model.json  
  |----train.json  
# Basic usage
  The first step you might want is to create a directory to dump the results.  
  `mkdir results`    
  `cd results`  
  `mkdir eval`  
  `mkdir grad_cam`  
  To train and test a model:  
  `python train_and_test.py --config-path config/ --model-path model/`  
  To predict the docking activity of a 60-nt pre-RNA:  
  `python predict.py --config-path config/ --model model/ --sequence [sequence]`  
  To interpret the model with Gradient Class Activation Mapping:  
  `python interpret.py --config-path config/ --model model/ --threshold [threshold score to select the candidates you are interested in]`
