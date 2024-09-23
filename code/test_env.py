import torch
import torch.nn as nn
import torch.optim as optim
import sklearn

print(sklearn.show_versions())

print(torch.__config__.show())

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print("Active CUDA device:", 
          torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("No active CUDA devices.")

help('modules')
