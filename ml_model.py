import numpy as np
import matplotlib.pyplot as plt
import utils.score as score
import matplotlib.pyplot as plt
import xgboost as xgb
#train_x = np.load("G:\\Business Analytics\\Information Retrival and Data Mining\\code for ir\\model_training_input.npy")
#train_y= np.load("G:\\Business Analytics\\Information Retrival and Data Mining\\code for ir\\model_training_output.npy")



def lin_reg(w,x):
    return w @ x

def lin_reg_loss(w,x,y):
    loss = np.sum((lin_reg(w,x) - y) **2 )
    return loss

def g_d(w,x,y,lr):
    gd = (y - lin_reg(w,x)) @ x.T
    w = w + lr * gd
    return w

def log_reg(w,x):
    z = w @ x
    return 1/(1 + np.exp(-z))

def log_reg_loss(w,x,y):
    loss = np.sum(y * np.log(log_reg(w,x)) + (1 - y) * np.log(1 - log_reg(w,x)))
    return np.abs(loss)

def g_d_log(w,x,y,lr):
    gd = (y - log_reg(w,x)) @ x.T
    w = w + lr * gd
    return w

#x = train_x.T
#y = train_y.T
#w = np.zeros((4,5))

####Linear Regression  ####
#loss = []
#w = np.zeros((4,5))
#for i in range(1000):
#    w = g_d(w,x,y,0.0000005)
#    yp = lin_reg(w,x)
#    loss.append(lin_reg_loss(w,x,y))
#loss1 = []
#w = np.zeros((4,5))
#for i in range(1000):
#    w = g_d(w,x,y,0.0000003)
#    yp = lin_reg(w,x)
#    loss1.append(lin_reg_loss(w,x,y))
#w = np.zeros((4,5))
#loss2 = []
#for i in range(1000):
#    w = g_d(w,x,y,0.0000001)
#    yp = lin_reg(w,x)
#    loss2.append(lin_reg_loss(w,x,y))
#i = np.arange(1000)
#plt.plot(i,loss,label = 'learning_rate 5e-7')
#plt.plot(i,loss1,label = 'learning_rate 3e-7')
#plt.plot(i,loss2,label = 'learning_rate 1e-7')
#legend()
#####Logstics Regression  ###
#loss1 = []
#for i in range(1000):
#    w = g_d_log(w,x,y,0.001)
#    yp = log_reg(w,x)
#    loss1.append(log_reg_loss(w,x,y))
#    
#ypp1 = np.zeros((4,49972))
#yp = logg
#
#for i in range(49972):
#    for j in range(4):
#        if yp[j][i] == np.max(yp[:,i]):
#            ypp1[j][i] = 1
#ypp = ypp.T
#ypp1 = ypp1.T

#r1= []
#r2 = []
#for i in range(len(y)):
#    r1.append(np.sum(y[i] == ypp[i]))
#    r2.append(np.sum(y[i] == ypp1[i]))

#for i in range(len(r1)):
#    if r1[i] == 4:
#        r1[i] = 1
#    else:
#        r1[i] = 0
#
#for i in range(len(r1)):
#    if r2[i] == 4:
#        r2[i] = 1
#    else:
#        r2[i] = 0
    
#linear_predict = lin_reg(w_linear_regression,model_test_input.T)
#ypp = np.zeros((4,25413))
#yp = linear_predict
#
#for i in range(25413):
#    for j in range(4):
#        if yp[j][i] == np.max(yp[:,i]):
#            ypp[j][i] = 1
#ypp = ypp.T
#
#
#
#logistic_predict = log_reg(w_linear_regression,model_test_input.T)
#ypp1 = np.zeros((4,25413))
#yp1 = logistic_predict
#
#for i in range(25413):
#    for j in range(4):
#        if yp1[j][i] == np.max(yp1[:,i]):
#            ypp1[j][i] = 1
#ypp = y_predict_linear_label.T
#ypp1 = y_predict_logistic_label.T
#
#y = model_training_output
#r1= []
#r2 = []
#for i in range(len(y)):
#    r1.append(np.sum(y[i] == ypp[i]))
#    r2.append(np.sum(y[i] == ypp1[i]))
#
#for i in range(len(r1)):
#    if r1[i] == 4:
#        r1[i] = 1
#    else:
#        r1[i] = 0
#
#for i in range(len(r1)):
#    if r2[i] == 4:
#        r2[i] = 1
#    else:
#        r2[i] = 0
#Pr = []   
#for i in range(len(ypp)):
#    if ypp[i][0] == 1:
#        a = 'agree'
#    if ypp[i][1] == 1:
#        a ='disagree'
#    if ypp[i][2] == 1:
#        a ='discuss'
#    if ypp[i][3] == 1:
#        a ='unrelated'
#    Pr.append(a)
#
#Pr1 = []   
#for i in range(len(ypp1)):
#    if ypp1[i][0] == 1:
#        a = 'agree'
#    if ypp1[i][1] == 1:
#        a ='disagree'
#    if ypp1[i][2] == 1:
#        a ='discuss'
#    if ypp1[i][3] == 1:
#        a ='unrelated'
#    Pr1.append(a)
#    
#Ac = []   
#for i in range(len(ypp)):
#    if y[i][0] == 1:
#        a = 'agree'
#    if y[i][1] == 1:
#        a ='disagree'
#    if y[i][2] == 1:
#        a ='discuss'
#    if y[i][3] == 1:
#        a ='unrelated'
#    Ac.append(a)
#
#ac = np.array(Ac)
#pr = np.array(Pr)
  
    
#############question 10  #########
#yp = np.zeros((25413,4))
#for i in range(4):
#    clf = xgb.XGBClassifier()
#    clf= clf.fit(trainX,trainY[:,i])
#    
#    valYY = clf.predict(testX)
#    valYP = clf.predict_proba(testX)[:,1]
#    for j in range(25413):
#        yp[j][i] = valYP[j]
        
#yp[:,0] = yp[:,0] - 0.07
#yp[:,1] = yp[:,1] - 0.02
#yp[:,2] = yp[:,2] - 0.17
#yp[:,3] = yp[:,3] - 0.74      
#yp = yp.T        
#ypp = np.zeros((4,25413))        
#for i in range(25413):
#    for j in range(4):
#        if yp[j][i] == np.max(yp[:,i]):
#            ypp[j][i] = 1
#ypp = ypp.T        
        
y = yt       
r1= []       
for i in range(len(y)):
    r1.append(np.sum(y[i] == ypp[i]))       
        
for i in range(len(r1)):
    if r1[i] == 4:
        r1[i] = 1
    else:
        r1[i] = 0        
        
#Pr = []   
#for i in range(len(ypp)):
#    if ypp[i][0] == 1:
#        a = 'agree'
#    if ypp[i][1] == 1:
#        a ='disagree'
#    if ypp[i][2] == 1:
#        a ='discuss'
#    if ypp[i][3] == 1:
#        a ='unrelated'
#    Pr.append(a)       
#        
#Ac = []   
#for i in range(len(ypp)):
#    if y[i][0] == 1:
#        a = 'agree'
#    if y[i][1] == 1:
#        a ='disagree'
#    if y[i][2] == 1:
#        a ='discuss'
#    if y[i][3] == 1:
#        a ='unrelated'
#    Ac.append(a)
#
#ac = np.array(Ac)
#pr = np.array(Pr)        
        
        
        
        
        
        

aha = score.report_score(Ac,Pr)   