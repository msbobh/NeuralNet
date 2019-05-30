function [J grad] = CostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%  CostFunction returns the gradient calculated for a neural network 
%  comprised of two layers. Resume classification model. The
%  second layers (hidden) has been optimized based on results from:
%  "How many hidden layers and nodes" D. Stathakis International journal
%  of Remote Sensing Vol. 30 No 8, 20, April 2009
% 
%   Calling conventions for this function:
%   [J grad] = CostFunction(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda)
%
%	J = Result of the const function calculation
%	grad = Uses gradient descent to optimize the cost function, grad returns the cost for each iteration
%   nn_params - The learned parameters for the Neural Network
%   hidden_layers - The number of neurons in the second layer of the network
%   number_labels - the number of labels
%   X - The training Matrix
%   y - Label vector
%   lambda - the size of the multiplier constant for regularization

%
% Initial Thetas are passed in as a single vector as needed by the fmiunc, reshape nn_params
% back into the parameters Theta1 and Theta2 for use in the const function


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1); % m equals the number of samples
  
%
% Initialize the return varialbles 
%
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


%   There are three components to this function:
%     Section 1:  calculates feedforward values
%     Section 2:  calculates the regularization terms and returns the error in J and
%                 the gradients in Theta1_grad and Theta2_grad
%     Section 3:  calculates the backprop
%


% If y has more than 1 column then this represents a multi class data set
if ( columns(y) > 1)
    yv = [1:num_labels] == y;
  else
    yv = y;
endif 

% 
% Part 1 - Feed forward calculations
%

% add a column of 1's to X matrix for bias terms
aone = [ones(m, 1) X];

% Vectorized calculation of the hypothesis over the layers

ztwo = aone * Theta1'; % m x hidden_layer_size
atwo = sigmoid (ztwo); % m x hidden_layer_size 
atwo = [ones(m,1) atwo]; % adding a column of 1's to m x hidden_layer_size + 1 

zthree = atwo * Theta2'; % m x num_labels   
hypothesis = sigmoid (zthree); % m x lables 

% Calculate Cost function over # samples m
J = (1/m) * sum(sum(( - yv .* log(hypothesis)) - ((1 - yv) .* log(1-hypothesis))));

% 
% Part 2 - Calculate regularization terms
%
% Note from regularization sum over the columns first k then j for Theta1 

J = J + lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2,2)) + sum(sum(Theta2(:,2:end).^2,2)));

delta = 0;
% Feed forward calculations for calculating the deltas

% Layer 1
a1 = [ones(m, 1) X]'; % aone already has the column of ones added in section 1

% Layer 2
z2 = Theta1 * a1; 
a2 = sigmoid(z2);  
a2 = [ones(m,1) a2']; 

% Layer 3
z3 =  a2 * Theta2'; % 5000 x 10
a3 = sigmoid(z3); %  5000 x 10

% 
% Part 3 - Calculate backprop 
%

% Calculate the deltas
delta3 =  a3 - y; % 1 
delta2 = (delta3 * Theta2) .* [ones(m,1) sigmoidGradient(z2)']; % delta2 5000 x 26
delta2 = delta2(:,2:end); % removes delta0 term 

Theta1_grad = Theta1_grad + (delta2' * a1'); % 25 x 401
Theta2_grad = Theta2_grad + (delta3' * a2); % 10 x 26



Theta1_grad =  (1/m) * Theta1_grad;
Theta2_grad =  (1/m) * Theta2_grad;

%
% Calculate the regularization for back propagation
%

Theta1_grad_reg_term = (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad_reg_term = (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];


Theta1_grad = Theta1_grad + Theta1_grad_reg_term;
Theta2_grad = Theta2_grad + Theta2_grad_reg_term;
keyboard;
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

fprintf('.'); % Write out a dot indicate training repetitions

end
