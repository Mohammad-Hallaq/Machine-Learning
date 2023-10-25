% ================================================================                 
%
%                             Lab 2
% 
% =================================================================
%% =========== Part 1a: Dimensionality reduction with PCA and LDA =========
clear all;

% PCA and LDA are very useful tools for plotting data that has more than 2
% dimensions. They can also be used for feature selection. 
% We will use the smallMNIST dataset to demonstrate how they work.
% This data set is used to classify handwritten digits 0-9. The
% file smallMNIST contains a smaller version of the popular MNIST data set.
% The variable X contains 5000 images (500 per digit). Each image is 20x20
% pixels big, so the total number of features per image is 400. The
% variable y is the labelvector where y(i) = k means instance i belongs to
% digit k except for k = 10 which represents the digit 0. 
load('smallMNIST.mat'); % Loads X and y

% Let's reduce the number of images to visualize the data better.
X = X(1:10:end,:);
y = y(1:10:end);

% ====================== YOUR CODE HERE ======================
% Use the function imagesc to plot the first image in this data set X(1,:).
% You can use the function reshape to reshape this vector of 400 elements
% to a 20 x 20 matrix.

imagesc(reshape(X(1,:),[20,20]));

% ============================================================

% First we can normalize the data with mean normalization. We can use the
% matlab function zscore which does the same thing as featureMeanNormalize.m 
% from lab1. 
X = zscore(X);

% Now we use the matlab pca function. Notice how PCA is an unsupervised
% algorithm and does not use the label vector y. 
[coefs, Xpca, variances] = pca(X);

% Plot the projected data in two dimensions. We only use the first two 
% columns in Xpca
plotData(Xpca(:,1:2), y)
title('PCA'); legend('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

% Next we use LDA on the same data. We will use an implementation of LDA 
% from the Matlab Toolbox for Dimensionality Reduction. Notice how lda also 
% takes the label vector y as input.
[Xlda, mapping] = lda(X, y);

% Plot the projected data from LDA.
plotData(Xlda(:,1:2), y);
title('LDA'); legend('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


%% ====== Part 1b: Principal Component Analysis (PCA): Reconstruction =====
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.
clear all;

% Load the data
load('smallMNIST.mat');

% Finish the code in mypca to calculate Xpca before continuing.
[coefs, Xpca, variances] = mypca(X);

% ====================== YOUR CODE HERE ===================================
% We can use Xpca and coefs to reconstruct an approximation of the data back
% to the original space using the top K eigenvectors in coefs. For the i-th 
% example Xpca(i,:), the (approximate) recovered data for dimension j is 
% given as: 
%                 Xrec(i,j) = Xpca(i, 1:K) * coefs(j, 1:K)';
%
% Xrec should have the same size as X.
 K =[2, 50, 400];
 for i=1:length(K)
     
      Xrec = Xpca(:, 1:K(i)) * coefs(:, 1:K(i))';
      figure(i)
      subplot(2,1,1)
      imagesc(reshape(X(1,:),[20,20]));
      subplot(2,1,2)
      imagesc(reshape(Xrec(1,:),[20,20]));
   
 end

% =========================================================================

%% ============= Part 2a: Clustering with GMM and k-means =================
clear all;

% We will use MATLABs GMM implementation for this part. First we load the
% data.
load simplecluster
X = zscore(X);

% Next we fit a GMM to the data
numberOfClusters = 3;
obj = gmdistribution.fit(X, numberOfClusters);

% Then classify each data point into one of the clusters
idx = cluster(obj,X);

% Plot the data and display the cluster number in each cluster center
plotData(X, idx); title(['GMM, #Clusters = ' num2str(numberOfClusters)]);

% ====================== YOUR CODE HERE ===================================
% Now do the same for k-means using the function kmeans. Type 
% >> help kmeans for more information how this function works. 
K = 3;
idx_Kmeans = kmeans(X,K);
plotData(X, idx_Kmeans); title(['Kmeans, #Clusters = ' num2str(K)]);

K = [3 4 5];
numberOfClusters = [3 4 5];

for i=1:length(K)
    
    obj = gmdistribution.fit(X, numberOfClusters(i));
    idx = cluster(obj,X);
    idx_Kmeans = kmeans(X,K(i));
    plotData(X, idx); title(['GMM, #Clusters = ' num2str(numberOfClusters(i))]);
    plotData(X, idx_Kmeans); title(['Kmeans, #Clusters = ' num2str(K(i))]);
    
end

% =========================================================================

%% ================ Part 2b: Clustering on smallMNIST data set ============
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.
clear all;
load('smallMNIST.mat'); % Loads X and y
X = zscore(X);
% ====================== YOUR CODE HERE ===================================
% Use K-means on the smallMNIST data set using K=10. Make a plot with 10 
% original images from each cluster.
K = 10;
idx = kmeans(X,K);
counter = -499;
for i=10:-1:1
    for j=1:length(idx)
        if idx(j) == i
            counter = counter + 500;
            figure;
            imagesc(reshape(X(counter,:),[20,20]));           
            break
        end
    end
end
% =========================================================================

%% =============== Part 3: Hidden Markov Model (HMM) ======================
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.

% Now we train a HMM to do temporal smoothing on predicted sleep stages.
clear all

load sleepdata % Gives xtrain, ytrain, xtest, ytest

% First we plot the data. Each sample is a segment of 5 seconds. The y
% values are the predicted sleep stage where 
% 5 = awake stage
% 4 = REM sleep
% 3 = stage 1 sleep
% 2 = stage 2 sleep
% 1 = deep sleep
% ytrain is the correct sleep stages labeled by a human expert. xtrain is a
% predicted sleep stage from a machine learning algorithm. 
figure; 
subplot(2,1,1); plot(ytrain, 'Color', 'r'); title('Correct sleep pattern');
subplot(2,1,2); plot(xtrain, 'Color', 'b'); title('Estimated sleep pattern');

% It can be seen that the predicted sleep stage (blue line) changes a lot.
% We know from ytrain that such a sleep pattern is not very probable. 

% We first train a HMM on this data and then used the trained HMM to smooth
% out the predictions from another patient. The predicted stage will be our
% observations and the hidden state is the true sleep stage. We will use 5
% hidden states, one for each sleep stage. TransitionMatrix(i,j) is the
% probability of transition from state i to j. EMIS(k,seq) is the probability 
% that symbol seq is emitted from state k
[TransitionMatrix, EmissionMatrix] = hmmestimate(xtrain, ytrain, ...
    'PSEUDOTRANSITIONS', 0.001*ones(5, 5), ...
    'PSEUDOEMISSIONS',0.001*ones(5, 5));

% ====================== YOUR CODE HERE ===================================
% Next we use the learned transition matrix and emission matrix to smooth
% out the predictions on a test patient. Use the function hmmviterbi to 
% calculate xtest_smooth from xtest,TransitionMatrix, and EmissionMatrix
xtest_smooth = hmmviterbi(xtest,TransitionMatrix,EmissionMatrix);

figure; 
subplot(3,1,1); plot(ytest, 'Color', 'r'); title('Correct sleep stage');
subplot(3,1,2); plot(xtest, 'Color', 'b'); title('Estimated sleep stage');
subplot(3,1,3); plot(xtest_smooth, 'Color', 'g'); title('Smoothed out sleep stage');

% =========================================================================

%% ===== Part 4a: Classification with Decision trees, k-NN, Naive Bayes ===
clear all;
load smallMNIST; % load X and y

% We first use a PCA to reduce the number of dimensions of the data
N = 10; % number of components to use
[coefs, Xpca, variances] = pca(X);
X = Xpca(:,1:N);

% We randomly split the data X and label vector y into 70% training
% data that we will use to train the classifiers and 30% testing data that
% we use to validate the model. Use the same random seed for X and y.
seed = 1;
[trainX, testX] = splitData(X, [0.7; 0.3], seed);
[trainy, testy] = splitData(y, [0.7; 0.3], seed);

% ====================== YOUR CODE HERE ===================================
% Use the functions ClassificationTree.fit, knnclassify, and 
% fitcnb to train a Decision Tree, k-NN classifier, and a
% Naive Bayes classifier on the training data and training labels trainX and
% trainy. Calculate the predictions using the function predict on the test 
% data testX and compute the classification accuracy using the test labels testy.
% HINT: The classfication accuracy can be computed by mean(y_pred(:)==y(:))
tree = ClassificationTree.fit(trainX, trainy);
Class = fitcknn(trainX,trainy,'NumNeighbors',5);
Mdl = fitcnb(trainX,trainy);
YPred_Tree = predict(tree,testX);
YPred_KNN = predict(Class,testX);
YPred_NBays = predict(Mdl,testX);

accuracy_Tree = mean(YPred_Tree(:)== testy(:));
accuracy_KNN =  mean(YPred_KNN(:)== testy(:));
accuracy_NBays =  mean(YPred_NBays(:)== testy(:));

K = [3 4 5 6 7 8];
N = [5 8 10 12 15 20];
for i=1:length(N)
    
    X = Xpca(:,1:N(i));
    seed = 1;
    [trainX, testX] = splitData(X, [0.7; 0.3], seed);
    [trainy, testy] = splitData(y, [0.7; 0.3], seed);
    tree = ClassificationTree.fit(trainX, trainy);
    Mdl = fitcnb(trainX,trainy);
    YPred_Tree = predict(tree,testX);
    YPred_NBays = predict(Mdl,testX);
    accuracy_Tree(i) = mean(YPred_Tree(:)== testy(:));
    accuracy_NBays(i) =  mean(YPred_NBays(:)== testy(:));
    
    for j=1:length(K)
       
        Class = fitcknn(trainX,trainy,'NumNeighbors',K(j));
        YPred_KNN = predict(Class,testX);    
        accuracy_KNN(i,j) =  mean(YPred_KNN(:)== testy(:));

    end
        
    
    
end

max_Tree_acc = max(accuracy_Tree);
max_NBays_acc = max(accuracy_NBays);
max_KNN_acc = max(max(accuracy_KNN));

% =========================================================================


%% ======== Part 4b: Classification with Classification Learner ===========
clear all;
load smallMNIST; % load X and y

seed = 1;
[trainX, testX] = splitData(X, [0.7; 0.3], seed);
[trainy, testy] = splitData(y, [0.7; 0.3], seed);

traindata = [trainX trainy];

classificationLearner

ypred = trainedModel.predictFcn(testX);
acc = mean(ypred == testy);


%% ================== Part 5: Spam Email Classification ==================
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.

%  Now we will classify emails into Spam or Non-Spam.
clear all;

% The file spamTrain.mat contains the feature vector for 4000 emails stored
% in a matrix X and the labels y. The file spamTest.mat contains 1000
% emails that we will use as test set. 
load('spamTrain.mat'); % Gives X, y to your workspace
load('spamTest.mat'); % % Gives Xtest, ytest to your workspace

% ====================== YOUR CODE HERE ======================
% Use the function svmTrain to train a Support Vector Machine (SVM). Set
% kernelFunction = @linearKernel and let tol and max_passes be the default
% values. Try different values for C (e.g. 1, 0.1, 0.01, 0.001). use
% svmPredict to calculate the classification accuracy on the test set. 
% OPTIONAL: You could also try the MATLAB implementation of 1-class SVM
% using the functions fitcsvm and predict. Try different options for the 
% parameter 'KernelFunction'.
C = [1, 0.1, 0.01, 0.001];
for i =1:length(C)
    
    model = svmTrain(X,y,C(i),@linearKernel);
    acc(i) = mean(svmPredict(model,Xtest)== ytest);
    
end
    disp (acc)
% ============================================================

% OPTIONAL:
% If you want to try your classifier on one of your own email you can copy 
% the email to emailSample.txt and convert the text to a feature vector
% using the code below. I put in one of my academic spam emails as default.
% Then feed the feature vector to your trained classifier (svmmodel). 

% Uncomment to run
% file_contents = readFile('emailSample.txt');
% word_indices  = processEmail(file_contents);
% features = emailFeatures(word_indices);
% predict(svmmodel, features)

