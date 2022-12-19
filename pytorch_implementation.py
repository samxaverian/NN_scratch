STUDENT_NAME = <STUDENT_NAME>  # Put your name
STUDENT_ROLLNO = <STUDENT_ROLLNO>  # Put your roll number
#@PROTECTED_1
##DO NOT MODIFY THE BELOW CODE. NO OTHER IMPORTS ALLOWED. NO OTHER FILE LOADING OR SAVING ALLOWED.
from torch import Tensor
import torch.nn as nn 
import torch.optim as optim 
import torchmetrics
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection 
import torch.utils.data as data 
import numpy as np 
import torch
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
submission = np.load("sample_submission.npy")
#@PROTECTED_1
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Model running on {}".format(device))
class NeuralNet(nn.Module):
    def __init__(self,input_features=3072,layer1=2000,layer2=1000,layer3=500,layer4=100,out_features=10):
        super().__init__()
        self.nn_stack = nn.Sequential(
            nn.Linear(3072,2000,bias=True),
            nn.ReLU(),
            nn.Linear(2000,1000,bias=True),
            nn.ReLU(),
            nn.Linear(1000,500,bias=True),
            nn.ReLU(),
            nn.Linear(500,100,bias=True),
            nn.ReLU(),
            nn.Linear(100,10)
        )
    def forward(self,x):
        z_output = self.nn_stack(x)
        return z_output

train_x = torch.from_numpy(X_train)
train_y = torch.from_numpy(y_train)
test_x = torch.from_numpy(X_test)
train_x = train_x.float()
train_y = train_y.type(torch.LongTensor)
test_x = test_x.float()

print("Transferring Datasets to GPU....")

train_x = train_x.to(device)
train_y = train_y.to(device)
test_x = test_x.to(device)

print("Transferring Model to GPU....")

model = NeuralNet().to(device)


print("...Training on train data...")
torch.manual_seed(100)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-3)
for i in range(500):
    y_pred = model.forward(train_x)
    loss = loss_function(y_pred,train_y)
    if(i%10==0):
        print("Epoch: {} ,Loss: {}".format(i,loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("...Predicting on test data...")
    
submission =[]
for i,x in enumerate(test_x):
    prob_x = model(x)
    submission.append(prob_x.argmax().item())

print("creating file")

#@PROTECTED_2
np.save("{}__{}".format(STUDENT_ROLLNO,STUDENT_NAME),submission)
print("prediction file created")
#@PROTECTED_2
