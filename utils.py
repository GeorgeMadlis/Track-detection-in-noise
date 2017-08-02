import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
from matplotlib import colors


def save_current_keras_model_json_hdf5(model, model_name):
    
    model_json = model.to_json()
    with open(model_name + '.json', 'w') as json_file:
        json_file.write(model_json)
    print('Saved model metadata to json format: ', model_name + '.json')

    model.save_weights(model_name + '.h5')
    print('Saved model to h5 format: ', model_name + '.h5')

def load_keras_model(model_name):
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_name + '.h5')
    print("Model loaded from file")
    return loaded_model  

def load_data(DATA_PATH, fname_start, no_files):
    
    X = []
    y = []

    for file_no in range(no_files):
        xfname = DATA_PATH+fname_start + '_X_' + str(file_no) + '.npy'
        yfname = DATA_PATH+fname_start + '_y_' + str(file_no) + '.npy'
        X1 = np.load(xfname)
        y1 = np.load(yfname)
        if len(X) == 0:
            X = X1
            y = y1
        else:
            X = np.vstack((X,X1))
            y = np.vstack((y,y1))
    return X, y

def roc_dat(estim_Y, Y, th):
        
    a_1 = np.zeros(estim_Y.shape[0])
    a_2 = np.zeros(estim_Y.shape[0])

    a = np.zeros(estim_Y.shape[0])
    b = np.zeros(estim_Y.shape[0])
    c = np.zeros(estim_Y.shape[0])
    d =  np.zeros(estim_Y.shape[0])
    d_1 =  np.zeros(estim_Y.shape[0])
    d_2 =  np.zeros(estim_Y.shape[0])

    count = -1
    for i in range(estim_Y.shape[0]):
        count += 1
        if estim_Y[i] >= th:
            a_2[count] = 1
        if Y[i] >= th:
            a_1[count] = 1
        if Y[i] < th:
            d_1[count] = 1
        if estim_Y[i] < th:
            d_2[count] = 1
        if estim_Y[i] >= th and Y[i] >= th:
            a[count] = 1 # true positive
        if estim_Y[i] < th and Y[i] >= th:
            b[count] = 1 # false negative
        if estim_Y[i] >= th and Y[i] < th:
            c[count] = 1 # false positive
        if estim_Y[i] < th and Y[i] < th:
            d[count] = 1 # true negative
    
    Pd = str(round(sum(a)/(sum(a)+sum(b)),3))
    Pfa = str(round(1 - sum(d)/(sum(c)+sum(d)),3))
    
    N_true_pos = str(round(sum(a_1),0))
    N_estim_pos = str(round(sum(a_2),0))
    N_true_neg = str(round(sum(d_1),0))
    N_estim_neg = str(round(sum(d_2),0))
    print('Pd: ' ,Pd, '# of true positives: ', N_true_pos, '# of estimated positives: ', N_estim_pos)
    print('P_false_pos_rate: ',  Pfa, '# of true negatives:  ', N_true_neg, '# of estimated negatives: ', N_estim_neg)
    # print('Error: ' , np.sum(np.abs(Y- estim_Y))/len(Y[:,0])*100)

def track_y_3D(y, n = 3):
    y_stacked = np.hstack((y,y))
    for i in range(n-2):
        y_stacked = np.hstack((y_stacked,y))
        
    return np.reshape(y_stacked,(y_stacked.shape[0],y_stacked.shape[1],1))    
   
def y_to_yplot(y, dx):
    
    y_plot = []
    for i in range(len(y)):
        y1 = np.ones((dx,1))*y[i]
        if len(y_plot) == 0:
            y_plot = y1
        else:
            y_plot = np.vstack((y_plot,y1))
    return np.array(y_plot).flatten()
    
def plot_results(test_y_true, Y_estim_test, train_X, train_y_true, Y_estim_train, test_X,
                 dxn, row1_train, row2_train, row1_test, row2_test, N_plots = 6):
    
    test_y_true_plot = y_to_yplot(test_y_true, dxn)
    test_y_estim_plot = y_to_yplot(Y_estim_test, dxn)

    y_plot =  y_to_yplot(train_y_true, dxn)
    train_y_estim_plot = y_to_yplot(Y_estim_train, dxn)

    fig=plt.figure(figsize=(16,20))
    cmap = colors.ListedColormap(['white','black'])

    
    plt.subplot(N_plots,1,1)
    plt.plot(y_plot[row1_train*dxn:row2_train*dxn],'bx-');
    plt.plot(train_y_estim_plot[row1_train*dxn:row2_train*dxn],'ro-');
    plt.title('y train subset')
    plt.subplot(N_plots,1,2)

    ims = train_X[row1_train:row2_train,:,:].copy()
    im = np.reshape(ims, (ims.shape[0]*ims.shape[1],ims.shape[2]))
    plt.imshow(im.transpose(),origin='lower', cmap=cmap, interpolation = 'none',aspect='auto');
    plt.title('Concatenated X train subset')
    #plt.xlabel('concatenated X column number')
    plt.ylabel('X row number')
    plt.subplot(N_plots,1,3)
    plt.plot(test_y_true_plot[row1_test*dxn:row2_test*dxn],'bx-');
    plt.plot(test_y_estim_plot[row1_test*dxn:row2_test*dxn],'ro-');
    plt.title('y test subset')
    plt.subplot(N_plots,1,4)
    ims = test_X[row1_test:row2_test,:,:].copy()
    im = np.reshape(ims, (ims.shape[0]*ims.shape[1],ims.shape[2]))
    plt.imshow(im.transpose(),origin='lower', cmap=cmap, interpolation = 'none',aspect='auto');
    plt.title('Concatenated X test subset')
    plt.xlabel('Concatenated X column number')
    plt.ylabel('X row number')
