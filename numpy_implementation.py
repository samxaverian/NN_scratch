STUDENT_NAME = <STUDENT_NAME>  # Put your name
STUDENT_ROLLNO = <STUDENT_ROLLNO>  # Put your roll number
CODE_COMPLETE = True
# set the above to True if you were able to complete the code
# and that you feel your model can generate a good result
# otherwise keep it as False
# Don't lie about this. This is so that we don't waste time with
# the autograder and just perform a manual check
# If the flag above is True and your code crashes, that's
# an instant deduction of 2 points on the assignment.
#
# @PROTECTED_1_BEGIN
## No code within "PROTECTED" can be modified.
## We expect this part to be VERBATIM.
## IMPORTS
## No other library imports other than the below are allowed.
## No, not even Scipy
import numpy as np
import pandas as pd
import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing
import sklearn.metrics as metrics
from tqdm import tqdm  # You can make lovely progress bars using this

## FILE READING:
## You are not permitted to read any files other than the ones given below.
X_train = pd.read_csv("train_X.csv", index_col=0).to_numpy()
y_train = (
    pd.read_csv("train_y.csv", index_col=0)
    .to_numpy()
    .reshape(
        -1,
    )
)
X_test = pd.read_csv("test_X.csv", index_col=0).to_numpy()
submissions_df = pd.read_csv("sample_submission.csv",index_col=0)
# @PROTECTED_1_END

y_train_onehot = np.zeros((y_train.shape[0], 10), dtype=np.float32)
y_train_onehot[y_train] = 1

ε = 1e-10


def ReLU(z):
    return np.clip(z, 0, z)


def Tanh(z):
    z = np.array(z, np.float128)
    z = np.clip(z, -11355, 11355)
    e_plus = np.exp(z)
    e_minus = np.exp(-z)
    return (e_plus - e_minus) / (e_plus + e_minus + ε)


def d_Tanh(z):
    z = np.array(z, np.float128)
    d = 1 - Tanh(z) * Tanh(z)
    return d


def Softmax(x):
    x = np.array(x, dtype=np.float128)
    x = np.clip(x, -11355, 11355)
    z = np.exp(x)
    return z / (np.sum(z, axis=1, keepdims=True) + ε)


def d_RelU(z):
    return (z > 0).astype(np.float32)


X_train = X_train / 255
mean = X_train.mean(axis=0)[None, :]
X_train = 2 * (X_train - mean)
# Xavier initialization
# weight initializing
limit1 = np.sqrt(6 / float(784 + 500))
limit2 = np.sqrt(6 / float(500 + 100))
limit3 = np.sqrt(6 / float(100 + 10))

w1 = np.random.uniform(low=-limit1, high=limit1, size=(784, 500))
w2 = np.random.uniform(low=-limit2, high=limit2, size=(500, 100))
w3 = np.random.uniform(low=-limit3, high=limit3, size=(100, 10))
w1 = np.array(w1, dtype=np.float128)
w2 = np.array(w2, dtype=np.float128)
w3 = np.array(w3, dtype=np.float128)
# bias initializing
limit4 = np.sqrt(6 / float(10000 + 500))
limit5 = np.sqrt(6 / float(10000 + 100))
limit6 = np.sqrt(6 / float(10000 + 10))
b1 = np.random.uniform(low=-limit4, high=limit4, size=(10000, 500))
b2 = np.random.uniform(low=-limit5, high=limit5, size=(10000, 100))
b3 = np.random.uniform(low=-limit6, high=limit6, size=(10000, 10))
b1 = np.array(b1, dtype=np.float128)
b2 = np.array(b2, dtype=np.float128)
b3 = np.array(b3, dtype=np.float128)

α = 1e-3

mini_batch_x = np.array(np.split(X_train, indices_or_sections=6, axis=0))
mini_batch_y = np.array(np.split(y_train_onehot, indices_or_sections=6, axis=0))
e_flag = 0
con = 0
epoch = 1
print("...Model Training...")
for i in range(100):
    batch = 1
    for batch_x, batch_y in zip(mini_batch_x, mini_batch_y):
        z1 = batch_x @ w1 + b1
        if np.isnan(z1).any():
            print("z1 has nan")
            e_flag = 1
            break
        if np.isinf(z1).any():
            print("z1 has inf")
            e_flag = 1
            break
        output1 = Tanh(z1)
        if np.isinf(output1).any():
            print("output1 has inf")
            e_flag = 1
            break
        if np.isnan(output1).any():
            print("output1 has nan")
            e_flag = 1
            break
        z2 = output1 @ w2 + b2
        if np.isinf(z2).any():
            print("z2 has inf")
            e_flag = 1
            break
        if np.isnan(z2).any():
            print("z2 has nan")
            e_flag = 1
            break
        output2 = Tanh(z2)
        if np.isinf(output2).any():
            print("output2 has inf")
            e_flag = 1
            break
        if np.isnan(output2).any():
            print("output2 has nan")
            e_flag = 1
            break
        z3 = output2 @ w3 + b3
        if np.isinf(z3).any():
            print("z3 has inf")
            e_flag = 1
            break
        if np.isnan(z3).any():
            print("z3 has nan")
            e_flag = 1
            break
        prob_x = Softmax(z3)
        if np.isinf(prob_x).any():
            print("prob_x has inf")
            e_flag = 1
            break
        if np.isnan(prob_x).any():
            print("prob_x has nan")
            e_flag = 1
            break

        loss = np.abs(batch_y * np.log(prob_x)) + np.abs(
            (1 - batch_y) * np.log(1 - prob_x + ε)
        )
        loss = loss.mean()
        if np.isnan(loss) or np.isinf(loss):
            e_flag = 1
            print("loss is nan or inf")
            break
        
        print("Epoch: {} , Batch: {} , Loss: {} ".format(epoch,batch,loss))
        batch = batch+1
        if loss <= 1e-2:
            con = 1
            print("Model converged significantly")
            break

        dl_dpx = batch_y / (prob_x + ε) - (1 - batch_y) / (1 - prob_x + ε)
        dpx_dz3 = prob_x * np.subtract(1, prob_x)
        dl_dz3 = dl_dpx * dpx_dz3  # element-wise
        dz3_dw3 = output2.T
        dl_dw3 = dz3_dw3 @ dl_dz3
        memo_1 = (dl_dz3 @ (w3.T)) * d_Tanh(z2)
        dl_dw2 = (output1.T) @ memo_1
        memo_2 = (memo_1 @ w2.T) * d_Tanh(z1)
        dl_dw1 = batch_x.T @ memo_2
        # balance the weights
        w3 = np.subtract(w3, -α * dl_dw3)
        w2 = np.subtract(w2, -α * dl_dw2)
        w1 = np.subtract(w1, -α * dl_dw1)
        # balancing the biases
        b3 = b3 + α * dl_dz3
        b2 = b2 + α * memo_1
        b1 = b1 + α * memo_2

    epoch = epoch + 1

    if e_flag:
        break
    if con:
        break

# Test predictions
print("...Predicting on test data...")
# test_batch = np.array(np.split(X_test, indices_or_sections=100, axis=0))
yp = []


tz1 = X_test @ w1 + b1
output1 = Tanh(tz1)
tz2 = output1 @ w2 + b2
toutput2 = Tanh(tz2)
tz3 = toutput2 @ w3 + b3
tprob_x = Softmax(tz3)
yp.append(np.argmax(tprob_x, axis=1))

print("Creating CSV file...")

ypred = np.array(yp)
ypred = ypred.flatten()
submissions_df = pd.DataFrame(ypred, columns=["label"])


# @PROTECTED_2_BEGIN
##FILE WRITING:
# You are not permitted to write to any file other than the one given below.
submissions_df.to_csv("{}__{}.csv".format(STUDENT_ROLLNO, STUDENT_NAME))
print("done ('J')")
# @PROTECTED_2_END
