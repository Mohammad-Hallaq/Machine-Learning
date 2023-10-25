% ================================================================                   
%
%                             Lab 3
% 
% ================================================================

addpath('./minFunc/'); %add minFunc to working directory after running the cd command 

%% ========= Part 1a: Regularization for Logistic Regression =============
% First we will add regularization to our previously written logistic
% regression. Copy costLogisticRegression.m and put it in the same folder 
% as this file (lab3.m).

clear all;

% Implement L2 weight decay regularization in costLogisticRegression.m. 
% Do not regularize the first element in theta. Check the gradients on a
% small randomized test data.
X = randn(10,10);
y = randi(2,10,1)-1;
initial_theta = randn(10,1);
lambda = 1;
[J, grad] = costLogisticRegression(initial_theta, X, y, lambda);
numgrad = checkGradient(@(p) costLogisticRegression(p, X, y, lambda), initial_theta);
diff = norm(numgrad-grad)/norm(numgrad+grad) % Should be less than 1e-9

% We will test if regularization can help an email spam detector. 
load spamTrain
X = [ones(size(X,1), 1) X]; % add intercept term

% The test set is located in spamTest.mat. We need to divide the
% training data into train and validation sets. 
[Xtrain, Xval] = splitData(X, [0.8; 0.2], 0);
[ytrain, yval] = splitData(y, [0.8; 0.2], 0);
initial_theta = zeros(size(X, 2), 1);
%clear X

% Train a logistic regression model on the full train set. We have
% provided two choices for optimization solver; minFunc and fmincg. You can 
% also use Matlab's fminunc or write your own similar to gradientDescent.m 
% from lab 1.
options = struct('display', 'on', 'Method', 'lbfgs', 'maxIter', 400);
lambda = 1e-4;
theta = fmincg(@(p)(costLogisticRegression(p, Xtrain, ytrain, lambda)), initial_theta, options);
%theta = minFunc(@(p)(costLogisticRegression(p, Xtrain, ytrain, lambda)), initial_theta, options);

% Calculate the classification accuracy on the train and validation set
trainaccuracy = mean(round(sigmoid(Xtrain*theta))==ytrain)
valaccuracy = mean(round(sigmoid(Xval*theta))==yval)

% The classification error is calculated as the 1-accuracy.
trainerror = 1 - trainaccuracy
valerror = 1 - valaccuracy

% ====================== YOUR CODE HERE ======================
% Make a plot of the classification accuracy or error on the train and 
% validation sets as a function of lambda. Try the following values for lambda:
% lambda_list = [0 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1]. You can use
% set(gca,'Xtick', 1:length(lambda_list), 'Xticklabel', lambda_list)
% to set the x-label as the values for lambda
lambda_list = [0 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1];
for i=1:length(lambda_list)
    
    theta = fmincg(@(p)(costLogisticRegression(p, Xtrain, ytrain, lambda_list(i))), initial_theta, options);
    trainaccuracy(i) = mean(round(sigmoid(Xtrain*theta))==ytrain);
    valaccuracy(i) = mean(round(sigmoid(Xval*theta))==yval);
end
figure;
plot(lambda_list, 1 - trainaccuracy)
title('Train error Vs lambda')
ylabel('Error')
xlabel('lambda')
hold on
plot(lambda_list,  1 - valaccuracy)
title('Validation error Vs lambda')
ylabel('Error')
xlabel('lambda')
set(gca,'Xtick', 1:length(lambda_list), 'Xticklabel', lambda_list)
legend('Train error','Validation error')

% ============================================================

% Load test set
load spamTest
Xtest = [ones(size(Xtest,1), 1) Xtest]; % add intercept term

% ====================== YOUR CODE HERE ======================
% Calculate the accuracy on the test set with the best choice for lambda
lambda = 1e-3;
theta = fmincg(@(p)(costLogisticRegression(p, Xtrain, ytrain, lambda)), initial_theta, options);
testaccuracy = mean(round(sigmoid(Xtest*theta))==ytest);

lambda = 0;
theta = fmincg(@(p)(costLogisticRegression(p, Xtrain, ytrain, lambda)), initial_theta, options);
testaccuracy_zerolambda = mean(round(sigmoid(Xtest*theta))==ytest);
% ============================================================

%% ================= Part 1b: n-fold cross-validation =====================
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.

% ====================== YOUR CODE HERE ======================
% Perform a 5-fold cross-validation on the training data and the optimal 
% choice of lambda from Part 1a. 
[X1, X2, X3, X4, X5] = splitData(X, [0.2; 0.2; 0.2; 0.2; 0.2], 0);
[y1, y2, y3, y4, y5] = splitData(y, [0.2; 0.2; 0.2; 0.2; 0.2], 0);

lambda = 1e-3;
for i=1:5
    
    X_groups = {X1, X2, X3, X4, X5};
    y_groups = {y1, y2, y3, y4, y5};
    X_test = X_groups{1,i};
    y_test = y_groups{1,i};
    X_groups{1,i} = [];
    y_groups{1,i} = [];
    X_train = vertcat(X_groups{:,:});
    y_train = vertcat(y_groups{:,:});
    theta = fmincg(@(p)(costLogisticRegression(p, X_train, y_train, lambda)), initial_theta, options);
    testaccuracy(i) = mean(round(sigmoid(X_test*theta))==y_test);
    
end

mean_acc = mean(testaccuracy);
std_acc = std(testaccuracy);

% ============================================================

%% ========= Part 2a: Logistic Regression for multiple classes =============
% In this part we will train a logistic regression classifier for the task
% of classifying handwritten digits [0-9]. 
clear all;

% First we load the data from the file smallMNIST.mat which is a reduced 
% set of the MNIST handwritten digit dataset. The full data set can be
% downloaded from http://yann.lecun.com/exdb/mnist/. Our data X consist of 
% 5000 examples of 20x20 images of digits between 0 and 9. The number "0" 
% has the label 10 in the label vector y. The data is already normalized.
load('smallMNIST.mat'); % Gives X, y

% We use displayData to view 100 random examples at once. 
[m, n] = size(X);
rand_indices = randperm(m);
figure; displayData(X(rand_indices(1:100), :));

% Now we divide the data X and label vector y into training, validation and
% test set. We use the same seed so that we dont get different
% randomizations. We will use hold-out cross validation to select the
% hyperparameter lambda.
seed = 1;
[Xtrain, Xval, Xtest] = splitData(X, [0.6; 0.3; 0.1], seed);
[ytrain, yval, ytest] = splitData(y, [0.6; 0.3; 0.1], seed);

% Now we train 10 different one vs all logistic regressors. Complete the
% code in trainLogisticReg.m before continuing. 
lambda = 0.01;
all_theta = trainLogisticReg(Xtrain, ytrain, lambda);

% Now we calculate the predictions using all 10 models. 
ypredtrain = predictLogisticReg(all_theta, Xtrain);
ypredval = predictLogisticReg(all_theta, Xval);
ypredtest = predictLogisticReg(all_theta, Xtest);
fprintf('Train Set Accuracy: %f\n', mean(ypredtrain==ytrain)*100) ;
fprintf('Validation Set Accuracy: %f\n', mean(ypredval==yval)*100);
fprintf('Test Set Accuracy: %f\n', mean(ypredtest==ytest)*100);

% It could be interesting to plot the missclassified examples.
% figure; displayData(Xtest(ypredtest~=ytest, :));

%% ========= Part 2b: Softmax classification for multiple classes ==========
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.

% In this part we will train a softmax classifier for the task of 
% classifying handwritten digits [0-9]. 
clear  all;

% Load the same data set. In softmax and neural networks the convention is 
% to let each column be one training input instead of each row as we have 
% previously used. 
load('smallMNIST.mat'); % Gives X, y
X = X'; y = y';

% Split into train, val, and test sets
seed = 1;
[Xtrain, Xval, Xtest] = splitData(X, [0.6 0.3 0.1], seed);
[ytrain, yval, ytest] = splitData(y, [0.6 0.3 0.1], seed);

% Initialize theta
numClasses = 10; % Number of classes
initial_theta = reshape(0.005 * randn(numClasses, size(X,1)), [], 1);

% For debugging purposes create a small randomized data matrix and
% labelvector. Calculate cost and grad and check gradients. Finish the code 
% in costSoftmax.m first. 
lambda = 1e-4;
[cost,grad] = costSoftmax(initial_theta, Xtrain(:,1:12), ytrain(1:12), numClasses, lambda);
numGrad = checkGradient( @(p) costSoftmax(p, Xtrain(:,1:12), ytrain(1:12), numClasses, lambda), initial_theta);
diff = norm(numGrad-grad)/norm(numGrad+grad) % Should be less than 1e-9

% Now we train the softmax classifier.
lambda = 0.01;
options = struct('display', 'on', 'Method', 'lbfgs', 'maxIter', 400);
theta = trainSoftmax(Xtrain, ytrain, numClasses, lambda, options);

% Now we calculate the predictions.
ypredtrain = predictSoftmax(theta, Xtrain, numClasses);
ypredval = predictSoftmax(theta, Xval, numClasses);
ypredtest = predictSoftmax(theta, Xtest, numClasses);
fprintf('Train Set Accuracy: %f\n', mean(ypredtrain==ytrain)*100) ;
fprintf('Validation Set Accuracy: %f\n', mean(ypredval==yval)*100);
fprintf('Test Set Accuracy: %f\n', mean(ypredtest==ytest)*100);

%% ============Part 2c: Plot Learning curve =====================
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.
clear 

load('smallMNIST.mat'); % Gives X, y
X = X'; y = y';

div_per = 4:0.2:8;
for i=1:length(div_per)
    
seed = 1;
[Xtrain, Xval, Xtest] = splitData(X, [div_per(i)*0.1 (9-div_per(i))*0.1 0.1], seed);
[ytrain, yval, ytest] = splitData(y, [div_per(i)*0.1 (9-div_per(i))*0.1 0.1], seed);

numClasses = 10; % Number of classes
initial_theta = reshape(0.005 * randn(numClasses, size(X,1)), [], 1);

lambda = 0.01;
options = struct('display', 'on', 'Method', 'lbfgs', 'maxIter', 400);
theta = trainSoftmax(Xtrain, ytrain, numClasses, lambda, options);

ypredtrain = predictSoftmax(theta, Xtrain, numClasses);
ypredval = predictSoftmax(theta, Xval, numClasses);

train_err(i) = 100 - mean(ypredtrain==ytrain)*100;
val_err(i) = 100 - mean(ypredval==yval)*100;


end
plot(div_per,train_err)
hold on
plot(div_per,val_err)
ylim([0,15])
ylabel('error')
xlabel('size of training data')
title('Learning curve')
legend('Jtrain','jvc')

%% ============== Part 3a: Implementing Neural network ====================
% Time to implement a non-regularized neural network.
clear all;

% Create a small randomized data matrix and labelvector for testing your 
% implementation.
X = randn(8, 100);
y = randi(10, 1, 100);

% Set Learning parameters. We start with coding the NN without any
% regularization
parameters = []; % Reset the variable parameters
parameters.lambda = 0; % Weight decay penalty parameter (not used in this exercise)
parameters.beta = 0; % sparsity penalty parameter (not used in this exercise)

% We initiliaze the network parameters assuming a small network of 8 input
% units, 5 hidden units, and 10 output units. 
[theta thetaSize] = initNNParameters(8, 5, 10);

% Calculate cost and grad and check gradients. Finish the code in 
% costNeuralNetwork.m first.
[cost,grad] = costNeuralNetwork(theta, thetaSize, X, y, parameters);
numGrad = checkGradient( @(p) costNeuralNetwork(p, thetaSize, X, y, parameters), theta);
diff = norm(numGrad-grad)/norm(numGrad+grad) % Should be less than 1e-9

%% ==== Part 3b: Neural network for handwritten digit classification ======
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.

clear all;

% Load the data set and split into train, val, and test sets.
load('smallMNIST.mat'); % Gives X, y
X = X'; y = y';
[Xtrain, Xval, Xtest] = splitData(X, [0.6 0.3 0.1], 0);
[ytrain, yval, ytest] = splitData(y, [0.6 0.3 0.1], 0);

% Set Learning parameters.
parameters = []; % Reset the variable parameters
parameters.lambda = 0; % This is a tunable hyperparameter (not used in this exercise)
parameters.beta = 0; % This is a tunable hyperparameter (not used in this exercise)
numhid = 10; % % This is a tunable hyperparameter

% Initiliaze the network parameters.
numvis = size(X, 1);
numout = length(unique(y));
[theta, thetaSize] = initNNParameters(numvis, numhid, numout);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) costNeuralNetwork(p, thetaSize, Xtrain, ytrain, parameters);

% Now, costFunction is a function that takes in only one argument (the 
% neural network parameters). Use tic and toc to see how long the training
% takes.
tic
options = struct('display', 'on', 'Method', 'lbfgs', 'maxIter', 400);
[optTheta, optCost] = minFunc(costFunction, theta, options);
toc

% fmincg takes longer to train. Uncomment if you want to try it.
% tic
% options = optimset('MaxIter', 400, 'display', 'on');
% [optTheta, optCost] = fmincg(costFunction, theta, options);
% toc

% You can visualize what the network has learned by plotting the weights of
% W1 using displayData.
[W1, W2, b1, b2] = theta2params(optTheta, thetaSize);
displayData(W1);

% Now we predict all three sets.
ypredtrain = predictNeuralNetwork(optTheta, thetaSize, Xtrain);
ypredval = predictNeuralNetwork(optTheta, thetaSize, Xval);
ypredtest = predictNeuralNetwork(optTheta, thetaSize, Xtest);

fprintf('Train Set Accuracy: %f\n', mean(ypredtrain==ytrain)*100);
fprintf('Val Set Accuracy: %f\n', mean(ypredval==yval)*100);
fprintf('Test Set Accuracy: %f\n', mean(ypredtest==ytest)*100);
%--------------------------------------------------------------------------

numhid = [4 6 8 10 12 14 16 18 20 22 24 26 28 30];
for i=1:length(numhid)
    
    [theta, thetaSize] = initNNParameters(numvis, numhid(i), numout);
    costFunction = @(p) costNeuralNetwork(p, thetaSize, Xtrain, ytrain, parameters);
    [optTheta, optCost] = minFunc(costFunction, theta, options);
    ypredtrain = predictNeuralNetwork(optTheta, thetaSize, Xtrain);
    ypredval = predictNeuralNetwork(optTheta, thetaSize, Xval);
    acc_train(i) = mean(ypredtrain==ytrain)*100;
    acc_val(i) = mean(ypredval==yval)*100;

    
end

plot(numhid, 100-acc_train)
hold on
plot(numhid, 100-acc_val)
title('bias-variance analysis')
xlabel('number of hidden layers')
ylabel('error')
legend('Trainn error', 'Validation error')


%% ============== Part 4a: Implementing Auto-encoder =======================
% Time to implement a non-regularized auto-encoder.
clear  all;

% Create a small randomized data matrix and labelvector
X = randn(8, 100);
y = randi(10, 1, 100);

% Set Learning parameters. 
parameters = []; % Reset the variable parameters
parameters.lambda = 0; % Weight decay penalty parameter (not used in this exercise)
parameters.beta = 0; % sparsity penalty parameter (not used in this exercise)

% We initiliaze the network parameters assuming a small network of 8 input
% units, 5 hidden units, and 8 output units (same as the number of input
% units).
[theta, thetaSize] = initAEParameters(8, 5);

% Calculate cost and grad and check gradients. Note how costAutoencoder.m 
% does not require the label vector y.
[cost,grad] = costAutoencoder(theta, thetaSize, X, parameters);
numGrad = checkGradient( @(p) costAutoencoder(p, thetaSize, X, parameters), theta);
diff = norm(numGrad-grad)/norm(numGrad+grad) % Should be less than 1e-9

%% ======= Part 4b: Reconstructing with Auto-encoder ===================
% Now we will plot reconstructions of the input data using an auto-encoder.
clear all;

load('smallMNIST.mat'); % Gives X, y
X = X'; y = y';
[Xtrain, Xval, Xtest] = splitData(X, [0.6 0.3 0.1], 0);
[ytrain, yval, ytest] = splitData(y, [0.6 0.3 0.1], 0);

% Set Learning parameters. 
parameters = []; % Reset the variable parameters
parameters.lambda = 0; % This is a tunable hyperparameter (not used in this exercise)
parameters.beta = 0; % This is a tunable hyperparameter (not used in this exercise)
numhid = 16; % This is a tunable hyperparameter
maxIter = 50; % This is a tunable hyperparameter

% Initiliaze the network parameters. Here we use initAEParameters.m
% instead.
numvis = size(X, 1);
[theta, thetaSize] = initAEParameters(numvis, numhid);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) costAutoencoder(p, thetaSize, Xtrain, parameters);

% Train the model
tic
options = struct('display', 'on', 'Method', 'lbfgs', 'maxIter', maxIter);
[optTheta, optCost] = minFunc(costFunction, theta, options);
toc

[W1, W2, b1, b2] = theta2params(optTheta, thetaSize);

figure;
h = sigmoid(bsxfun(@plus, W1*Xtrain, b1)); %hidden layer
Xrec = sigmoid(bsxfun(@plus, W2*h, b2)); % reconstruction layer
subplot(1,2,1); displayData(Xtrain(:,1:100)'); title('Original input')
subplot(1,2,2); displayData(Xrec(:,1:100)'); title('Reconstructions')


%% ======= Part 4c: Classification with Auto-encoder ===================
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.

% ====================== YOUR CODE HERE ======================
% Use the trained auto-encoder from part 4b and use
% a classifier of your choice (e.g., trainLogisticReg or trainSoftmax) and
% calculate the classification accuracy on the test set.
lambda = 0.01;
options = struct('display', 'on', 'Method', 'lbfgs', 'maxIter', 400);
numClasses = 10;

theta = trainSoftmax(h, ytrain, numClasses, lambda, options);
h_test = sigmoid(bsxfun(@plus, W1*Xtest, b1)); %hidden layer

ypredtest = predictSoftmax(theta, h_test, numClasses);
fprintf('Test Set Accuracy: %f\n', mean(ypredtest==ytest)*100);
% ============================================================

