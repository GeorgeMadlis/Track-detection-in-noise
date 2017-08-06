The main file with details: TD_TNN.ipynb notebook

Uses pre-calculated binary 3D training and test images. These images are calculated in five steps:

    Calculate an image M with an uniform noise with the size of 30x30 pixels
    Calculate a track using linear regression formula y = a*x + b, where a and b are respectively the randomly generated slope and bias. Here x changes from 0 to 30 at a predefined step dx; -1 <= a <= 1; 10 <= b <= 90
    Map the track obtained in step 2 to M, such that 0 <= y <= 30.
    Reshape the M 30x30 image into a 3D matrix M1 with the size 1,10,30
    Repeat either steps 1 to 4 or steps 1,4 (i.e. create a noise image only). Choose randomly between two loops i.e. choose between steps 1-4 or steps 1-5. Stack M1 matrices, so that the repeating steps 1-4 N times, we get a M1 with the size N*3,10,30

As a result we get a 3D image data matrix where each image M1(i,:,:) includes either noise only or track in noise. Depending on the sampled values of a and b, a single track may or may not extend through the three consecutive images M(i,j,:), M(i,j+1,:), and M(i,j+2,:). Track presence/absence of the track is labeled respectively by y(i) = 1 and y(i) = 0

The aim of this test is to train a RNN which is able to detect presence of tracks in noisy data.

Besides TD_TNN.ipynb, this repo includes the following data files:

    train_linear_X_0.npy - 3D matrix M1 used for training. Each M1(i,:,:) is labeled by:
    train_linear_y_0.npy - 1D vector.
    test_linear_X_0.npy - 3D matrix M1 used for testing.
    test_linear_y_0.npy - vector used for labeling of test data M(i,:,:).
    keras_3k_dat_linmodel.h5 - trained LSTM model weights
    keras_3k_dat_linmodel.json - json file defining LSTM model architecture used in this work
