function theta = trainSoftmax(X, y, numClasses, lambda, options)
%softmaxTrain Train a softmax model with the given data and labels. Returns 
% the wight matrix theta.
%
% X: an n by m matrix containing the input data, such that X(:, c) is the 
% cth input
% y: m by 1 matrix containing the class labels for the corresponding 
% inputs. labels(c) is the class label for the cth input
% options: (optional) Can change the number of iterations

if ~exist('options', 'var')
    options = struct;
end

if ~isfield(options, 'maxIter')
    options.maxIter = 400;
end

% initialize parameters
initial_theta = reshape(0.005 * randn(numClasses, size(X, 1)), [], 1);

% Here we use minFunc to minimize the function. But we could have use any 
% of minFunc, fmincg, fminunc, or gradientDescent.m from lab 1
theta = minFunc( @(p) costSoftmax(p, X, y, numClasses, lambda), initial_theta, options);

% Unroll theta
theta = theta(:);
                          
end                          
