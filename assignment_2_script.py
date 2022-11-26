STUDENT_NAME = yourNameHere #Put your name
STUDENT_ROLLNO = yourRollNumberHere #Put your roll number
CODE_COMPLETE = False 
# set the above to True if you were able to complete the code
# and that you feel your model can generate a good result
# otherwise keep it as False
# Don't lie about this. This is so that we don't waste time with
# the autograder and just perform a manual check
# If the flag above is True and your code crashes, that's
# an instant deduction of 2 points on the assignment.
#
#@PROTECTED_1_BEGIN
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
from tqdm import tqdm # You can make lovely progress bars using this

## FILE READING: 
## You are not permitted to read any files other than the ones given below.
X_train = pd.read_csv("train_X.csv",index_col=0).to_numpy()
y_train = pd.read_csv("train_y.csv",index_col=0).to_numpy().reshape(-1,)
X_test = pd.read_csv("test_X.csv",index_col=0).to_numpy()
submissions_df = pd.read_csv("sample_submission.csv",index_col=0)
#@PROTECTED_1_END




#@PROTECTED_2_BEGIN 
##FILE WRITING:
# You are not permitted to write to any file other than the one given below.
submissions_df.to_csv("{}__{}.csv".format(STUDENT_ROLLNO,STUDENT_NAME))
#@PROTECTED_2_END