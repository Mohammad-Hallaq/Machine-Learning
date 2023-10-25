function [theta thetaSize] = initAEParameters(numvis, numhid)
% Initializes the parameters for an auto-encoder. 
% numvis - number of visible units (input units)
% numhid - number of hidden units

%We initialize the biases b_i to zero, and the weights W^i to random 
% numbers drawn uniformly from the interval 
% [ -sqrt(6/(n_in+n_out+1)) , sqrt(6/(n_in+n_out+1)) ]
% where n_in is the fan-in (the number of inputs feeding into a node) and 
% nout is the fan-in (the number of units that a node feeds into). This
% initiliazation is better than randomly drawn values around 0

r1  = sqrt(6) / sqrt(numvis+numhid+1);
r2  = sqrt(6) / sqrt(numhid+numvis+1);
W1 = rand(numhid, numvis)*2*r1 - r1;
W2 = rand(numvis, numhid)*2*r2 - r2;
b1 = zeros(numhid, 1);
b2 = zeros(numvis, 1);

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];
thetaSize = [size(W1); size(W2); size(b1); size(b2)];

end

