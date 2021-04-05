import numpy as np
import scipy.io as io

def SaveAsMat (traningSetID, testingSetID, pred, label):
    datasetName = ['cvd', 'liveq', 'livev','kon']
    traningSetName = datasetName[traningSetID]
    testingSetName = datasetName[testingSetID]
    mat_path = traningSetName+'_'+testingSetName+'.mat'
    pred, label = np.asarray(pred).squeeze(), np.asarray(label).squeeze()     
    io.savemat(mat_path, {'pred': pred, 'mos': label})
    print(mat_path+' have saved!')
    

if __name__ == "__main__":
    
    traningSetName = 0
    testingSetName = 3
    pred = np.zeros([20])
    label = np.ones([20])
    SaveAsMat (traningSetName, testingSetName, pred, label)