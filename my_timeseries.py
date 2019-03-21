import numpy as np
import pandas as pd
def predict_time_series(data,num_weeks):
    pr=num_weeks*7*24*4
    ldatetime=data['date_time'].iloc[-1]
    dti = pd.date_range(ldatetime, periods=pr, freq='.25H')

    #create training samples (features)
    num_weeks_train=20
    num_tr_samples=num_weeks_train*7*24*4
    ldatetime=data['date_time'].iloc[-1]
    nm_predic_smpl=pr
    trdt=data['total_calls'].iloc[-num_tr_samples:-1]
    dim=4*24
    trdt=trdt.values
    vsize=trdt.size
    X_tr=np.array([np.zeros(dim)]);Y_tr=np.zeros(1);
    for i in range(vsize-dim):
        x_tr=[trdt[i:i+dim]]
        y_tr=trdt[i+dim]
        X_tr=np.append(X_tr,x_tr,axis=0)
        Y_tr=np.append(Y_tr,y_tr)

    #Seasenality data preparation
    X_tr_se=np.zeros(X_tr.shape)
    for i in range(X_tr.shape[0]):
        xr=X_tr[i,:]
        xr0=putone(xr,i)
        X_tr_se[i,:]=xr0

    #Seasonal Regression
    coefb=(np.linalg.inv(X_tr_se.T@X_tr_se))@np.dot(X_tr_se.T,Y_tr)
    testset=data['total_calls'].iloc[-dim:].values
    stidx=X_tr.shape[0]+1
    predval=[]
    for i in range(nm_predic_smpl):
        testsetz=putone(testset,stidx+i)
        pred=np.dot(coefb,testsetz)
        predval=np.append(predval,pred)
        testset=np.append(testset,pred)
        testset=testset[-dim:]


    d={'date_time': pd.Series(dti),'total_calls':pd.Series(predval)}
    predictdata = pd.DataFrame(d)
    return predictdata

def putone(x,loct):
    nm_pt_day=4*24 #number of points in a day
    slt=loct%nm_pt_day
    s=np.zeros(nm_pt_day)
    s[slt]=1
    return np.multiply(x,s)
    
