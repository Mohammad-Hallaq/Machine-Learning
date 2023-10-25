function [theta J_history] = gradientDescent_mom(theta, X, y, alpha, num_iters)
% Runs gradient descent.

gamma = 0.5;
velocity = 0;

J_history = zeros(num_iters, 1);
for iter = 1:num_iters
    [J, grad] = costLinearRegression(theta, X, y);
    J_history(iter) = J;
    
    % ====================== YOUR CODE HERE ======================
    % Update the parameter vector theta by using alpha and grad.
    if iter > 5
        
        gamma = 0.9;
    end
     velocity = gamma*velocity + alpha.*grad;
     theta = theta - velocity;
    
    % ============================================================
    
end

end
