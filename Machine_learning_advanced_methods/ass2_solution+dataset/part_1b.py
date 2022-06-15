import csv
from libsvm.svmutil import *
import numpy as np
import random
import time

start = time.time()
def randomj(m, i):
    while 1==1:
        j = random.randint(0, m-1)
        if(j!=i):
            break
    return j

def cal_accuracy(y_t, beta, w, x_t):
    accu =0
    for a in range(len(x_t)):
    	accu= accu + predicted_model_for_pve(x_t[a], beta, w,y_t[a][0])
    return accu


def predicted_model_for_pve(ps,b, w,y):
    p = np.array([ps])
    f_val = np.dot(p,w.T) + b
    if ((f_val[0][0]>0 and y>0) or (f_val[0][0]<0 and y<0)):    #sahi hnkya bhai?? ->na bero toh pato laga				
        return 1
    else:
        return 0

def predicted_model_for_negve(ps,b, w): 
    p = np.array([ps])
    f_val = np.dot(p, w.T)+b
    if f_val[0][0]<0:
        return 1
    else:
        return 0


def find_w(alphas, x_t, y_t):
  #  m ,n = shape(x_t)
    new_x = np.array(x_t)
    new_y = np.array(y_t)
    m, n = len(new_x), len(new_x[0])
    w = np.zeros((1, n))# zeroes((n,1))
    bet = np.multiply(alphas, new_y)
    trans_bet = bet.T  
    return np.dot(trans_bet, new_x)
    
    
def foundsui(aj,H,L):
      if aj > H:
            aj = H
      if L > aj:
            aj = L
      return aj

def simplifiedSMO(dataX, classY, C, tol, max_no_of_passes):          # C is basically the regularization parameter
    X = np.array(dataX)
    Y = np.array(classY)
    m,n = len(X),len(X[0])
    beta =0
    alphas = np.zeros((m,1))
    passes = 0
    while passes<max_no_of_passes:
        num_chng_al = 0
        for i in range(m):
            fir_term = np.multiply(alphas, Y)
            trans_fir_term = fir_term.T
            ############################################## fill this
            sec_term = np.dot(X, X[i:i+1,:].transpose())
            trans_sec_term = sec_term
            fXi = np.dot(trans_fir_term , trans_sec_term ) + beta
            if fXi[0][0]>0:
                Ei = 1- Y[i][0]
            else:
                Ei = -1 - Y[i][0]
            if ((Y[i]*Ei < -tol) and (alphas[i][0] < C)) or ((Y[i]*Ei > tol) and (alphas[i][0] > 0)):
                j = randomj(m,i)
                fir_term_j = np.multiply(alphas, Y)
                trans_fir_term_j = fir_term_j.T
                ############################################## fill this
                sec_term_j = np.dot(X, X[j:j+1,:].transpose())
                trans_sec_term_j = sec_term_j
                fXj = np.dot(trans_fir_term_j , trans_sec_term_j ) + beta
               # fXj = float(multiply(alphas,Y).T*(X*X[j,:].T)) + beta
                if fXj[0][0]>0:
                    Ej = 1- Y[j][0]
                else:
                    Ej = -1 - Y[j][0]
                alphaIold = alphas[i][0]
                alphaJold = alphas[j][0]
                if (Y[i] != Y[j]):
                    L = max(0, alphas[j][0] - alphas[i][0])
                    H = min(C, C + alphas[j][0] - alphas[i][0])
                else:
                    L = max(0, alphas[j][0] + alphas[i][0] - C)
                    H = min(C, alphas[j][0] + alphas[i][0])
                if L==H:
                    continue
                
		
                ########################################################### update this    
                eta = 2.0 * np.dot(X[i,:],X[j,:].T) - np.dot(X[i,:],X[i,:].T) - np.dot(X[j,:],X[j,:].T)   
                if eta >= 0:
                    continue
                           
                alphas[j][0] -= float(Y[j]*(Ei - Ej))/eta
                alphas[j][0] = foundsui(i ,H,L) # are yo shyd rhgo
                if (abs(alphas[j][0] - alphaJold) < 0.00001):
                    continue

                alphas[i] += Y[j]*Y[i]*(alphaJold - alphas[j])

                ##############################################################update these
                b1 = beta - Ei- Y[i]*(alphas[i]-alphaIold)*np.dot(X[i,:],X[i,:].T)- Y[j]*(alphas[j]-alphaJold)*np.dot(X[i,:],X[j,:].T)
                b2 = beta - Ej- Y[i]*(alphas[i]-alphaIold)*np.dot(X[i,:],X[j,:].T )- Y[j]*(alphas[j]-alphaJold)*np.dot(X[j,:],X[j,:].T)

                if (0 < alphas[i][0]) and (C > alphas[i][0]):
                    beta = b1
                elif (0 < alphas[j][0]) and (C > alphas[j][0]):
                    beta = b2
                else:
                    beta = (b1 + b2)/2.0                   
                num_chng_al += 1
        if (num_chng_al == 0): 
            passes += 1
        else: 
            passes = 0
    return alphas,beta


####wra s dekh haa bero s phele SMO dekhe tho
read = csv.reader(open('2019CS10355.csv'), delimiter=',')
fir_class = input("Enter the first of the two classes: ")
sec_class = input("Enter your second class: ")
tole = float(input("Enter the numerical tolerance you want: "))
C = float(input("Enter the regularization parameter you want: "))
max_no_of_passes = int(input("THE max number of times to iterate over lang. multipliers: "))

my_class = []
my_features = []

if int(fir_class) >=int(sec_class):
    for row in read:
        if (float(row[25]) == float(fir_class)):
            my_class.append([1])
            my_features.append([float(num) for num in row[0:25]])
        if (float(row[25]) == float(sec_class)):
            my_class.append([-1])
            my_features.append([float(num) for num in row[0:25]])
else:
    for row in read:
        if (float(row[25]) == float(fir_class)):
            my_class.append([-1])
            my_features.append([float(num) for num in row[0:25]])
        if (float(row[25]) == float(sec_class)):
            my_class.append([1])
            my_features.append([float(num) for num in row[0:25]])

#print(len(my_class))
per_data_to_divide = int(0.80* len(my_class))
#print (per_data_to_divide)
my_class_newtest = my_class[per_data_to_divide:]
#print(len(my_class_newtest))
my_features_newtest = my_features[per_data_to_divide:]
my_class_newtrain = my_class[0:per_data_to_divide]
#print(len(my_class_newtrain))
my_features_newtrain = my_features[0:per_data_to_divide]

alpha, beta = simplifiedSMO(my_features_newtrain, my_class_newtrain, C, tole, max_no_of_passes)
w = find_w(alpha, my_features_newtrain, my_class_newtrain)

fin_acc = cal_accuracy(my_class_newtest,beta, w, my_features_newtest)
print(f'Our Approx. Accuracy is  {fin_acc*100.00/len(my_class_newtest)}')
end = time.time()
print(f'The time taken by the algorithm is {end-start}')



