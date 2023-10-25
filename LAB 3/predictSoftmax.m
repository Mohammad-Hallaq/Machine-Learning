function pred = predictSoftmax(theta, X, numClasses)
% theta - model trained using trainSoftmax
% X - the n x m input matrix, where each column X(:, i) corresponds to a 
% single test set
% pred - m x 1 vector with predicted class where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = reshape(theta, numClasses, size(X,1));

% Compute pred using theta and X assuming that the labels start from 1
h = theta*X;
h = bsxfun(@minus, h, max(h, [], 1)); % Preventing overflows 
h = exp(h);
h = bsxfun(@rdivide, h, sum(h));
[~, pred] = max(h,[],1);

% Make sure pred is a column vector
pred = pred(:)';

end

