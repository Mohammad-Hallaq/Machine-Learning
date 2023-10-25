function [coefs, Xpca, variances] = mypca(X)

[m, n] = size(X);
[coefs, variances] = svd(1/m*(X'*X));
variances = diag(variances);

% ====================== YOUR CODE HERE ======================
% Compute the projection of the data Xpca using the component coefficients, coefs
% and the input data X. For the i-th example X(i,:), the projection on to the k-th 
% eigenvector is given as follows: 

Xpca = X * coefs;
% ============================================================

end
