import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from pyntcloud import PyntCloud
import random
import seaborn as sns
import tqdm
from typing import List

from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from utils import PointCloudDataSet, SamplePoints, TREE_SPECIES
from model import PointNet

data_folder = "/scratch/projects/workshops/gpu-workshop/synthetic_trees_full_resolution/"

saved_image_path = "./saved_images"
if not os.path.exists(saved_image_path):
    os.makedirs(saved_image_path)
    print(f"Created path for images in {saved_image_path}")

transformations = transforms.Compose([SamplePoints(1024, sample_method = "random")])
# transformations = transforms.Compose([SamplePoints(1024, sample_method = "farthest_points")])

data = PointCloudDataSet(data_folder, train=False, transform=transformations)

ex_idx = 103
print(len(data))
points = data[ex_idx]["points"]
label = data[ex_idx]["label"]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], 'gray')
plt.title(f"How the nerual networks sees {label}")
save_image_in_file = 'scatter_points.png'
plt.savefig(os.path.join(saved_image_path, save_image_in_file))

#########################
###### Inference ########
#########################

dataloader = DataLoader(data, batch_size=8, shuffle=False, num_workers = 8)
n_classes = len(TREE_SPECIES)

 # Load model.
model = PointNet(n_classes)
# If you would like to use a trained model, otherwise comment out the next two lines.
load_model_path = "./model_test.pt"
model.load_state_dict(torch.load(load_model_path, map_location=torch.device('cpu')))
print(model)

model.eval(); # Layers will be in evaluation mode (e.g., no drop out).

 # Now we evaluate all 2000 trees.

all_true_labels = []
all_predictions = []

with torch.no_grad(): # Deactivates the autograd engine needed for the backward pass -> speedup.
    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        labels = batch["label"]
        true_labels = [TREE_SPECIES.index(l) for l in labels]
        points = batch["points"]

        predictions = model(points) # Points input batch size x sample size (1024) x dimensions (3) 
        predicted_labels, _, _ = predictions # Predictions are classes, embedding3x3, embedding64x64
        predicted_labels = torch.argmax(predicted_labels, 1)
        
        all_predictions.extend(predicted_labels)
        all_true_labels.extend(true_labels)

cm = confusion_matrix(all_true_labels, all_predictions) # row: true label, col: predicted label
print(cm)

fig, axs = plt.subplots()
df_cm = pd.DataFrame(cm, TREE_SPECIES, TREE_SPECIES)
s = sns.heatmap(df_cm, annot=True, fmt='d')
s.set(xlabel = "Predicted Species", ylabel = "True Species")
plt.title("Confusion Matrix for True vs Predicted Species")
plt.tight_layout()
save_image_in_file = 'confusion_matrix.png'
plt.savefig(os.path.join(saved_image_path, save_image_in_file))
#plt.savefig('confusion_matrix.png')

# Overall accuracy.
accuracy = np.sum(np.diagonal(cm))/ np.sum(cm)
print(f"The accuracy is {accuracy * 100}%.")