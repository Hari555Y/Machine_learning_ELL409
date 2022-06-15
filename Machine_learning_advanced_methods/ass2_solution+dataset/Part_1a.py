import csv
from libsvm.svmutil import *
import numpy as np
import random
import time

start = time.time()

read = csv.reader(open('2019CS10355.csv'), delimiter=',')
my_class = []
my_features = []

redu1 =[]
redu2 = []

model_type = input("Press 0 for binaryClass and 1 for multiclass classification :")
no_of_features = input("Press 10 or 25 for respective no of features :")
if int(no_of_features) ==25:
    if int(model_type)==0:
        fir_class = input("Enter the first of the two classes :")
        sec_class = input("Enter your second class :")
        for row in read:
            if (float(row[25]) == float(fir_class)) or (float(row[25]) == float(sec_class)):
                my_class.append(float(row[25]))
                my_features.append([float(num) for num in row[0:25]])
            else:
                redu1.append(float(row[25]))
                redu2.append([float(num) for num in row[0:25]])
    else:
        for row in read:
            my_class.append(float(row[25]))
            my_features.append([float(num) for num in row[0:25]])
else:
    if int(model_type)==0:
        fir_class = input("Enter the first of the two classes :")
        sec_class = input("Enter your second class :")
        for row in read:
            if (float(row[25]) == float(fir_class)) or (float(row[25]) == float(sec_class)):
                my_class.append(float(row[25]))
                my_features.append([float(num) for num in row[0:10]])
            else:
                redu1.append(float(row[25]))
                redu2.append([float(num) for num in row[0:10]])
    else:
        for row in read:
            my_class.append(float(row[25]))
            my_features.append([float(num) for num in row[0:10]])

per_data_to_divide = int(0.80* len(my_class))
#print (per_data_to_divide)
my_class_newtest = my_class[per_data_to_divide:]
my_features_newtest = my_features[per_data_to_divide:]
my_class_newtrain = my_class[0:per_data_to_divide]
my_features_newtrain = my_features[0:per_data_to_divide]


problem = svm_problem(my_class, my_features)
para = svm_parameter("-q")
gam = input("Enter the value of variable gamma: ")
kery = input("Enter the type of kernel you want to use: ")
para_valC = input("Enter the value of parameter C you want :")
#str = ""
ker_type = int(kery)
if ker_type==1:
    degree_poly = input("Enter the degree of polynomial you want: ")  
'''if ker_type ==0:
    str = "LINEAR"
elif ker_type == 1:
    str = "POLY"
elif ker_type ==2:
    str = "RBF"
else:
    str = "SIGMOID"'''
para.C = int(para_valC)
para.gamma = float(gam)
para.kernel_type = ker_type
if ker_type ==1:
    para.degree = int(degree_poly)
para.cross_validation =1
para.nr_fold = 10
fir_model= svm_train(problem, para)

#print (f"Training Accuracy is {accuracy}")
para.cross_validation = 0
problem = svm_problem(my_class_newtrain, my_features_newtrain)
train_model = svm_train(problem,para)
Y_predicted,Accu,Prob = svm_predict(my_class_newtest,my_features_newtest,train_model)
end = time.time()
print(f'time taken by this algorithm is {end-start}')
#print (f"Training Accuracy is {accuracy}")
