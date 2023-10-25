function [all_theta] = trainLogisticReg(X, y, lambda)
% Trains multiple logistic regression classifiers and returns all
% the classifiers in a matrix all_theta, where the i-th column of all_theta 
% corresponds to the classifier for label i vs all the rest

% Some useful variables
num_labels = length(unique(y));
[m n] = size(X);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Set Initial theta
initial_theta = zeros(n + 1, 1);

% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 50);

% You need to return the following variables correctly 
all_theta = zeros(n + 1, num_labels);

% ====================== YOUR CODE HERE ======================
% Complete the following code to train num_labels logistic regression 
% classifiers with regularization parameter lambda. You can use y == c to 
% obtain a vector of 1's and 0's that tell us whether the ground truth is 
% true/false for this class. Use a for-loop to loop to train a logistic 
% regressor classifier for each class. To train a logistic regressor that 
% separates class 1 vs all the other classes:
% Note: This function will be very slow if you don't have a vectorized
% version of costLogisticRegression.m

for i=num_labels:-1:1
    
    all_theta(:,i) = fmincg(@(p)(costLogisticRegression(p, X, y == i, lambda)), initial_theta, options);
    
end


% =========================================================================


end
