import numpy as np
import pandas as pd
#np.set_printoptions(threshold=np.nan)
def predict_time_series(data,num_weeks):
    pr=num_weeks*7*24*4
    ldatetime=data['date_time'].iloc[-1]
    dti = pd.date_range(ldatetime, periods=pr, freq='.25H')

    #create training samples (features)
    num_weeks_train=5
    num_tr_samples=num_weeks_train*7*24*4
    ldatetime=data['date_time'].iloc[-1]
    nm_predic_smpl=pr
    trdt=data['total_calls'].iloc[-num_tr_samples:]
    dim=7*4*24
    trdt=trdt.values
    vsize=trdt.size
    Y_tr=np.zeros([]);
    for i in range(vsize-dim):
        x_tr=[trdt[i:i+dim]]
        y_tr=trdt[i+dim]
        Y_tr=np.append(Y_tr,y_tr)

    #Seasenality data preparation
    X_tr_se=np.zeros((Y_tr.shape[0],(24*4)+6))
    for i in range(Y_tr.shape[0]):
        
        j=np.floor(i/(24*4))
        xr0=putone(i,j)
        X_tr_se[i,:]=xr0

    
    #Seasonal Regression
        
  #  coefb=(np.linalg.inv(X_tr_se.T@X_tr_se))@np.dot(X_tr_se.T,Y_tr)
    coefb=np.linalg.lstsq(X_tr_se,Y_tr,rcond=None)[0]
   # print(coefb,coefb.shape)
    testset=data['total_calls'].iloc[-dim:].values
    stidx=Y_tr.shape[0]
    predval=[]
    for i in range(nm_predic_smpl):
        obidxt=stidx+i
        obidxd=np.floor(obidxt/(24*4))
        testsetz=putone(obidxt,obidxd)
   #     if i==0:
   #         print(X_tr_se,testsetz)
        pred=np.dot(coefb,testsetz)
        predval=np.append(predval,pred)


    d={'date_time': pd.Series(dti),'total_calls':pd.Series(predval)}
    predictdata = pd.DataFrame(d)
    return predictdata

def putone(loct,locd):
    nm_pt_day=4*24 #number of points in a day
    slt=loct%nm_pt_day
    dslt=locd%6
    s=np.zeros(nm_pt_day+6)
    s[slt]=1
    secidx=nm_pt_day+dslt
    secidx=secidx.astype(int)
    s[secidx]=1
    return s
    
