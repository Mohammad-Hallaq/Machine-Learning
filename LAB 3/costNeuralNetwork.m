function [J, grad] = costNeuralNetwork(theta, thetaSize, X, y, parameters)
% Returns cost and gradient vector for a 3-layered Neural Network.
% theta - A column vector of the parameters W,b. The structure of theta is:
% theta = [W1(:); W2(:); b1(:) b2(:)]
% thetaSize - A 4x2 matrix where thetaSize(1,1) is the number of rows
% of W1 and thetaSize(1,2) is the number of columns in W1. thetaSize(2,:)
% is the number of rows and columns of W2 etc.
% X - the n x m input matrix, where each column X(:, i) corresponds to one
% training example
% y - an 1xm vector containing the labels corresponding for the input data
% parameters - a structure of all the parameters

% Useful parameters
m = size(X, 2);
numClasses = length(unique(y));

% Reshape theta to the network parameters
[W1, W2, b1, b2] = theta2params(theta, thetaSize);

% Get the parameters
lambda = parameters.lambda; % Weight decay penalty parameter
beta = parameters.beta; % sparsity penalty parameter
p = 0.05; % sparsity activation parameter rho. This can actually also be 
% considered a tunable parameter but we set this to 0.05.

% Convert row vector y to a binary matrix
y = full(sparse(y, 1:m, 1));
%y = full(sparse(1:m, y, 1, m, numClasses))'; %alternative

% You should compute the following variables
J = 0;
gradW1 = zeros(size(W1));
gradW2 = zeros(size(W2));
gradb1 = zeros(size(b1));
gradb2 = zeros(size(b2));

% ---------- YOUR CODE HERE --------------------------------------
% Compute the cost and gradient for neural networks. Do this step by step.
%
% (1): Perform the feed forward calculations to get the variables z2, a2, 
% z3, and a3. Use the function bsxfun and sigmoid (included in this file as 
% a subfunction, check the end of this file). Use the variables W1 W2 b1 b2

 z2 = bsxfun(@plus, W1*X, b1);
 a2 = sigmoid(z2);
 z3 = bsxfun(@plus, W2*a2, b2);
 a3 = sigmoid(z3);

% (2): Calculate the cost function, J. Use the squared error cost function.

 J = sum(sum((a3 - y).^2))/(2*m);

% (3): Calculate the error terms delta3 and delta2. For the calculation
% of delta2 use .* and sigmoidGradient(z2) (another function provided at
% the end of this file)

 delta3 = -(y - a3).*sigmoidGradient(z3);
 delta2 = (W2'*delta3).*sigmoidGradient(z2);

% (4): Use delta3 and delta2 to calculate gradW2, gradW1, gradb1, and
% gradb2. Here you also need to use delta3, delta2, a2, a1 (a1=X), and m. 
% Then you are ready to check your gradient with the analytical gradients.

gradW2 = (1/m)*delta3*a2';
gradW1 = (1/m)*delta2*X';
gradb1 = (1/m)*sum(delta2,2);
gradb2 = (1/m)*sum(delta3,2);


% ------------------------------------------------------------------

% Unroll the gradient matrices into a vector for the gradient method
grad =  [gradW1(:); gradW2(:); gradb1(:); gradb2(:)];

end

% Some helping functions that you might use. These can only be called 
% inside this function (costNeuralNetwork.m)
function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
    z = sigmoid(z);
    g = z.*(1-z);
end

function sigm = sigmoid(z)
% SIGMOID return the output from the sigmoid function with input z
sigm = 1./(1 + exp(-z));
end



