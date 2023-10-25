function theta = normalEqn(X, y)
%Computes the closed-form solution to linear regression.

theta = pinv(X'*X)*X'*y;

end
