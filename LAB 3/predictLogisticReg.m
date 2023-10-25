function p = predictLogisticReg(all_theta, X)

m = size(X, 1);

% You need to return the following variables correctly 
p = zeros(m, 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Calculate the predictions
[~,p] = max(sigmoid(all_theta'*X'),[],1);

% Unroll to make sure p is a row vector
p = p(:);

end

