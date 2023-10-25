% =================================================================
%
%                             Lab 1
% 
% =================================================================
%% ======================= Part 1a: Plotting =======================
load carbig.mat
% This command loads a number of variables to the MATLAB workspace. The 
% variables contain information about cars such as WEIGHT, Horsepower, 
% Model_Year e.t.c. The task is to fit a linear model that predicts the 
% horsepower based on the weight of the car.

x = Weight; % data vector
y = Horsepower; % prediction values

% The first thing to do when working with a new data set is to plot it. For 
% this data set you don't want to draw any lines between each data point so 
% set the plot symbol to point "." Type ">> help plot" to see how.
% ====================== YOUR CODE HERE ======================

 plot(x,y,'.')

% ============================================================

xlabel('Weight [kg]');
ylabel('Horsepower [hp]');

%% ======================= Part 1b: Remove NaN values ==============
% Next we need to clean up the data and remove any training data that
% contains any NaN (not-a-number) or Inf (infinity) values. 

% Complete the code in RemoveData.m
[x y] = RemoveData(x, y);

%% ======================= Part 1c: Normalize data ==============
% Scale the features down using mean normalization.
% Complete the code in featureMeanNormalize.m
[x mu sigma] = featureMeanNormalize(x);

%% ======= Part 2: Linear Regression: Cost function and derivative ========
% In this part we will implement the linear regression algorithm.

% Load the data
clear;
load carbig.mat
x = Weight; % data vector
y = Horsepower; % prediction values
[x y] = RemoveData(x, y);
[x mu sigma] = featureMeanNormalize(x);

X = [ones(size(x,1), 1), x]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Complete the code in costLinearRegression.m before continuing.
[J grad] = costLinearRegression(theta, X, y);

% You can check if your solution is correct by calculating the numerical
% gradients (numgrad) and compare them with the analytical gradients 
% (grad).
numgrad = checkGradient(@(p) costLinearRegression(p, X, y), theta);

% If your implementation is correct the two columns should be very similar. 
disp([numgrad grad]); 
% and the diff below should be less than 1e-9
diff = norm(numgrad-grad)/norm(numgrad+grad) 

%% =================== Part 3a: Gradient descent ===================
% In this part we will implement gradient descent and train our linear
% regression model.

clear;
load carbig.mat
x = Weight; % data vector
y = Horsepower; % prediction values
[x y] = RemoveData(x, y);
[x mu sigma] = featureMeanNormalize(x);

X = [ones(size(x,1), 1), x]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Hyperparameters for gradient descent 
num_iters = 1500;
alpha = 0.01;

% Run Gradient Descent. Complete the code in gradientDescent.m.
[theta J_history] = gradientDescent(theta, X, y, alpha, num_iters);

% Plot J_history. If your implementation is correct J should decrease after
% each iteration.
figure;
plot(J_history);
xlabel('Iteration')
ylabel('Cost J(\theta)')

% Plot the data and the linear regression model
figure;
plot(X(:,2), y, 'b.'); hold on; 
plot(X(:,2), X*theta, 'r-')
legend('Training data', 'Linear regression')

%% ========= Part 3b: Gradient descent with momentum ==================
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.

% ====================== YOUR CODE HERE ======================
% Implement momentum in gradientDescent.m. Set the gamma variable to 0.5
% for the first 5 iterations and to 0.9 after that. Plot the cost as a
% function of the iterations (J_history) for both when momentum is used and
% when it is not used for the same data as part 3a.

[theta_mom J_history_mom] = gradientDescent_mom(theta, X, y, alpha, num_iters);
[theta J_history] = gradientDescent(theta, X, y, alpha, num_iters);


figure;
plot(J_history_mom);
hold on
plot(J_history);
xlabel('Iteration')
ylabel('Cost J(\theta)')
legend('momentum is used', 'momentum is not used')

figure;
plot(X(:,2), y, 'b.'); hold on; 
plot(X(:,2), X*theta_mom, 'r-')
legend('Training data', 'Linear regression')
% ============================================================

%% ======= Part 4a: Linear regression with multiple variables =============
% In this part we are going to change the code in the following files so 
% that your implementation of Linear Regression works for multiple 
% variables:
%       RemoveData.m
%       featureMeanNormalize.m
%       costLinearRegression.m

% Load data
clear;
load carbig.mat
x = [Weight MPG]; % We use two variable - weight and miles per gallon (MPG)
y = Horsepower; % prediction values

% Plot the data. Use Tools -> Rotate 3D to examine the data.
figure;
plot3(x(:,1),x(:,2),y,'.','Color','b')
xlabel('Weight [kg]');
ylabel('Fuel efficiency [MPG]');
zlabel('Horsepower [hp]');
grid on

% Remove pairs of data that contains any NaN values. Change the code in
% RemoveData.m so that it works for multiple variables if needed.
[x y] = RemoveData(x, y);

% Normalize both feature vectors. Change the code in featureMeanNormalize.m 
% so that it works for multiple variables if needed.
[x mu sigma] = featureMeanNormalize(x);

X = [ones(size(x,1), 1) x]; % Add intercept term to X
theta = zeros(3, 1);

% Check gradients. Change the code in costLinearRegression.m so that it 
% works for multiple variables if needed.
[J grad] = costLinearRegression(theta, X, y);
numgrad = checkGradient(@(p) costLinearRegression(p, X, y), theta);
diff = norm(numgrad-grad)/norm(numgrad+grad) % Should be less than 1e-9

% Hyperparameters for gradient descent 
alpha = 0.1;
num_iters = 500;

% Run Gradient Descent
[theta J_history] = gradientDescent(theta, X, y, alpha, num_iters);

% Plot J_history. If your implementation is correct J should decrease after
% each iteration.
plot(J_history)

% Plot the data and the linear regression model
plot3(X(:,2), X(:,3), y, 'b.'); hold on; 
range=-2:.1:2;
[xind,yind] = meshgrid(range);
Z = zeros(size(xind));
for i=1:size(xind,1)
    for j=1:size(xind,2)
        Z(i,j) = [1 xind(i,j) yind(i,j)]*theta;
    end
end
surf(xind,yind,Z)
shading flat; grid on;
xlabel('Normalized Weight'); 
ylabel('Normalized MPG'); 
zlabel('Horsepower [hp]')
legend('Training data', 'Linear regression')

% ====================== YOUR CODE HERE ======================
% Predict how much horsepower a car would have that weights 3000 kg and has 
% a MPG of 30. Hint: It should be around 98.
% TIPS: Remember that you have to normalize the values first using mu and sigma.
  
  x_pred = [3000 30];
  x_pred = (x_pred - mu)./sigma;
  y_pred = [1 x_pred]*theta;

% ============================================================


%% ===================== Part 4b: Vectorization ===========================
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.


%% ================ Part 5: Normal Equation ===============================
% In this part we will use the normal equation to calculate the values for
% theta instead of using gradient descent.

% Load data. This time it is not necessary to normalize the data. 
clear;
load carbig.mat
x = [Weight MPG]; % We use two variable - weight and miles per gallon (MPG)
y = Horsepower; % prediction values
[x y] = RemoveData(x, y); % Remove bad training data
X = [ones(size(x,1), 1) x]; % Add intercept term to X

% Calculate the theta parameters with the normal equation. 
theta = normalEqn(X, y);

% ====================== YOUR CODE HERE ======================
% Predict how much horsepower a car would have that weights 3000 kg and has 
% a MPG of 30 using theta calculated from the normal equation. You should 
% get almost the same answer as in Part 4.

  x_pred = [3000 30];
  y_pred_normalEqn = [1 x_pred]*theta;

% ============================================================


%% ==================== Part 6a: Logistic Regression ======================
% In this part we will implement the logistic regression classification
% algorithm.

% Load data. This data set contain the information about age, sex, weight, and blood
% pressure for 100 patients. The task is to train a logistic regression
% classifier to classify whether a patient is a smoker or a non-smoker.
clear; % Clear all workspace variables
load hospital.mat
x = [hospital.Age hospital.BloodPressure(:,1)]; %We start with two input features - age and blood pressure
y = hospital.Smoker; % Label vector. 1 = Smoker, 0 = Non-smoker

% Plot the data
plot(x(y==1,1), x(y==1,2), 'b+'); hold on;
plot(x(y==0,1), x(y==0,2), 'ro')
legend('Smoker', 'Non-smoker')
xlabel('Age'); ylabel('Blood Pressure')

% Implement the sigmoid function. Complete the code in sigmoid.m. The 
% function should perform the sigmoid function of each element in a vector
% or matrix. 
g = sigmoid([-10 0.3; 0 10])
% If your implementation is correct you should get the following answer.
% g =
% 
%     0.0000    0.5744
%     0.5000    1.0000

% Add intercept term to x and initialize theta
[m, n] = size(x);
X = [ones(size(x,1), 1) x];
initial_theta = zeros(n + 1, 1);

% Now it is time to implement logistic regression in
% costLogisticRegression.m. Complete the code in costLogisticRegression.m
% before continuing.
[J grad] = costLogisticRegression(initial_theta, X, y);

% You can check if your implementation is correct by comparing the
% gradients with the analytical gradients.
numgrad = checkGradient(@(p) costLogisticRegression(p, X, y), initial_theta);
diff = norm(numgrad-grad)/norm(numgrad+grad) % Should be less than 1e-9

% Instead of using our own implementation of gradient descent 
% (gradentDescent.m) we will use the pre-built MATLAB function fminunc
% which sets the learning rate alpha automatically.
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
[theta, cost] = fminunc(@(t)(costLogisticRegression(t, X, y)), initial_theta, options);

% Plot data and decision boundary
plot(X(y==1,2), X(y==1,3), 'b+'); hold on
plot(X(y==0,2), X(y==0,3), 'ro');
plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
plot(plot_x, plot_y, 'Color', 'k', 'Linewidth', 2)
xlabel('Age'); ylabel('Blood Pressure')
legend('Smoker', 'Non-smoker')

% Now we use the learned logistic regression model to predict the 
% probability that a patient with age 32 and blood pressure 124 is a 
% smoker. 
prob = sigmoid([1 32 124] * theta) % Should return a value around 0.35
% So the probability that this patient is a smoker (the positive class y=1)
% is 35% and the probability that this patient is a non-smoker is
% 1-prob = 65%. This means that we classify this patient as a non-smoker.
% However, if you look at the plot it looks like this is a
% missclassification because the training data contains many smokers around
% that age and blood pressure. It seems like the assumption of a line is
% not enough to capture enough variations in the data.


%% ======== Part 6b: Logistic Regression with multiple variables ==========
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.

% In this part we introduce polynomial features in order to fit a curve
% instead of a line to the data. You might need to change
% costLogisticRegression.m so that it works for more than two variables.

% Load data
clear; % Clear all workspace variables
load hospital.mat
x = [hospital.Age hospital.BloodPressure(:,1)];
y = hospital.Smoker; % Label vector. 1 = Smoker, 0 = Non-smoker
[m, n] = size(x);
X = [ones(size(x,1), 1) x];

% Create polynomial features
degree = 2;
Xpoly = mapFeature(X(:,2), X(:,3), degree);
% Feature normalization becomes important when using polynomial features
[Xpoly(:,2:end) mu sigma] = featureMeanNormalize(Xpoly(:,2:end)); 
initial_theta = zeros(size(Xpoly, 2), 1);
lambda = 0; % regularization parameter. Ignore this for now, we will use it in later labs.

% Set options and optimize theta
options = optimset('GradObj', 'on', 'MaxIter', 100);
[theta, J, exit_flag, output] = fminunc(@(t)(costLogisticRegression(t, ...
    Xpoly, y, lambda)), initial_theta, options);

% Plot data and Boundary
plot(X(y==1,2), X(y==1,3), 'b+'); hold on
plot(X(y==0,2), X(y==0,3), 'ro');
u = linspace(15, 60, 50);
v = linspace(95, 150, 50);
z = zeros(length(u), length(v));
for i = 1:length(u)
    for j = 1:length(v)
        temp = (mapFeature(u(i), v(j), degree) - [1 mu])./[1 sigma];
        z(i,j) = sigmoid(temp*theta);
    end
end
z = z'; % important to transpose z before calling contour
contour(u, v, z, [0.5 0.5], 'Color', 'k', 'LineWidth', 2)

% ====================== YOUR CODE HERE ======================
% Calculate the probability that a patient with age 32 and blood pressure 
% 124 is a smoker.
% HINT: Remember to create the polynomial features for this data point and 
% normalize the data and use sigmoid.

x_pred = [1 32 124];
temp_pred = (mapFeature(x_pred(2), x_pred(3), degree) - [1 mu])./[1 sigma];
prob = sigmoid(temp_pred*theta);
% ============================================================
