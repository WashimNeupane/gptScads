-Install all dependencies manually(to be added soon). Some essential dependencies. 
   .peft
   .torch
   .datasets
   .tqdm

-For preprocessing dataset, run the preprocess.py scripts. It should generate a data folder with train and validation set of data. 

-Run the train.py file. The arguments it takes are -c (for peft configs), -t (for training arguments) and -m (for method i.e. peft, distillation, llm training). The arguments are read inside config folder stored in respective subdirectories. 
 
