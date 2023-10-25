function [J, grad] = costSoftmax(theta, X, y, numClasses, lambda)
% theta - A row vector of the parameters
% X - the n x m input matrix, where each column X(:, i) corresponds to one
% training example
% y - an 1 x m matrix containing the labels corresponding for the input data
% numClasses - number of classes
% lambda - weight decay parameter

% Get number of input units and training examples
[n m] = size(X);

% Resize the parameters of the row vector theta into a matrix
theta = reshape(theta, numClasses, n);

% Convert row vector y to a marix that represents the indicator
% function 1{y(i)=j}. 
y = full(sparse(1:m, y, 1, m, numClasses))';
%y = full(sparse(y, 1:m, 1));

% Calculate h (also written as p( y(i) = j | x(i); theta))
h = theta*X;
h = bsxfun(@minus, h, max(h, [], 1)); % Preventing overflows 
h = exp(h);
h = bsxfun(@rdivide, h, sum(h));

% You should return the following variables
J = 0;
grad = zeros(size(theta));

% ---------- YOUR CODE HERE --------------------------------------
% Compute the cost and gradient for softmax regression with weight decay 
% regularization. You will need to use h, y, theta, X, lambda, and m.


 J = (-1/m)*sum(sum(y.*log(h))) + (lambda/2)*sum(sum(theta.^2));
 grad = (-1/m)*(y-h)*X' + lambda*theta;

% ------------------------------------------------------------------

% Unroll the gradient matrices into a vector for the gradient method
grad = grad(:);

end


