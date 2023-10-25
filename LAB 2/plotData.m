function plotData(X, y)
% Plots the 2-dimensional matrix X. Each data point of the same class has
% the same color/marker. X is a m-by-2 matrix and y is a m-by-1 vector.

K = length(unique(y)); %number of classes
colors = {'r.' 'g.' 'b.' 'k.' 'rx' 'gx' 'bx' 'kx' 'ro' 'go' 'bo' 'ko'};

figure; hold on;

% Plot the data in X with different colors according to the label vector y.
for i=1:K
    plot(X(y==i,1), X(y==i,2), colors{i})
end

