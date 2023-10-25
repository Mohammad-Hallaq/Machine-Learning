function [J, grad] = costLogisticRegression(theta, X, y, lambda)
% Compute cost and gradient for logistic regression.

if nargin<4
    lambda = 0;
end

m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Compute the cost (J) and partial derivatives (grad) of a particular 
% choice of theta. Make use of the function sigmoid.m that you wrote earlier.

for i=1:m
    
    J = J + y(i)*log(sigmoid(X(i,:)*theta))+(1-y(i))*log(1-sigmoid(X(i,:)*theta));
    
end

J = -1*J/m;

for i=1:m
    
    for j=1:length(theta)
        
        grad(j) = grad(j)+ (sigmoid(X(i,:)*theta)- y(i))*X(i,j);
    end
   
end

grad = grad./m;

% =============================================================

% unroll gradients
grad = grad(:);

end
