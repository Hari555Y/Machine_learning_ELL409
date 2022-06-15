import csv
from libsvm.svmutil import *
import numpy as np
import random
read_train = csv.reader(open('train_set.csv'), delimiter=',')
my_class_train = []
my_features_train = []
for row in read_train:
    my_class_train.append(float(row[25]))
    my_features_train.append([float(num) for num in row[0:25]])
        
read_test = csv.reader(open('test_set.csv'), delimiter=',')
my_class_test = []
my_features_test = []
for row in read_test:
    my_class_test.append(0.0)
    my_features_test.append([float(num) for num in row[0:25]])	

problem = svm_problem(my_class_train, my_features_train)
para = svm_parameter("-q")
para.cross_validation =1
gam = input("Enter the value of variable gamma: ")                           #default value is 1/num_of_features       
kery = input("Enter the type of kernel you want to use: ")                   #default value is 2
thepara_C = input("Enter the value of C for the SVM: ")                      #default value is 1
para.nr_fold = 10
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
para.C = int(thepara_C)
para.gamma = float(gam)
para.kernel_type = ker_type
if ker_type ==1:
    para.degree = int(degree_poly)
fir_model= svm_train(problem, para)
#print (f"Training Accuracy is {accuracy}")
para.cross_validation = 0
problem = svm_problem(my_class_train, my_features_train)
train_model = svm_train(problem,para)
Y_predicted,Accu,Prob = svm_predict(my_class_test,my_features_test,train_model)


with open('try_1.csv', 'w+') as my_file:
    csv.writer(my_file).writerow(['Id', 'Class'])
    for i in range(len(Y_predicted)):
        mystr = str(i+1)
        if i+1>=1000:
            csv.writer(my_file).writerow([mystr[0]+','+mystr[1:],Y_predicted[i]])
        else:
            csv.writer(my_file).writerow([mystr, Y_predicted[i]])

'''with open('try_1.csv', 'w+') as my_file:
    my_file.write('Id, Class\n')
    for i in range(len(Y_predicted)):
        if i+1<1000:
            my_file.write('{}, {:d}\n'.format(str(i+1), int(Y_predicted[i])))
        else:
            my_file.write('\"{:01d},{:03d}\",{:d}\n'.format((i+1)//1000, (i+1)%1000, int(Y_predicted[i])))'''
    