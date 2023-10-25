function [theta J_history] = gradientDescent(theta, X, y, alpha, num_iters)
% Runs gradient descent.

J_history = zeros(num_iters, 1);
for iter = 1:num_iters
    [J, grad] = costLinearRegression(theta, X, y);
    J_history(iter) = J;
    
    % ====================== YOUR CODE HERE ======================
    % Update the parameter vector theta by using alpha and grad.
    
     theta = theta - alpha.*grad;
    
    % ============================================================
    
end

end
