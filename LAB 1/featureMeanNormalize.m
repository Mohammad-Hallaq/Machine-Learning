function [X_norm, mu, sigma] = featureMeanNormalize(X)
%Normalizes the features in X with mean normalization
% X_norm is the normalized version of X where each column has 0 mean and 
% standard deviation 1.
% mu and sigma are vectors of the mean and standard deviation for each
% feature column in X.

% You need to return the following variables correctly.
mu = zeros(1,size(X,2));
sigma = zeros(1,size(X,2));
X_norm = zeros(size(X));

% ====================== YOUR CODE HERE ======================
% Each column in X is a feature vector. The number of columns in X is the 
% number of features. For each feature compute the mean and standard
% deviation of that feature. Then subtract each feature vector with the
% mean and then divide by the standard deviation. Use the functions mean
% and std.

 mu = mean(X);
 sigma = std(X);
     
 X_norm = (X-mu)./sigma;

% ============================================================

end
