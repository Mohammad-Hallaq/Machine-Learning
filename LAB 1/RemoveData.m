function [x y] = RemoveData(x,y)

% You might find this vector useful
rowsToBeRemoved = find(sum((isnan([x y])),2)+sum(isinf([x y]),2)~=0);

% ====================== YOUR CODE HERE ======================
% Removes any rows in x and y that contain any NaN or Inf values. 
% Remember that you have to remove the same rows in both x and y.
% HINT: You can use the function removerows or x(rowsToBeRemoved,:) = [];

x(rowsToBeRemoved,:) = [];
y(rowsToBeRemoved,:) = [];

% ============================================================

end