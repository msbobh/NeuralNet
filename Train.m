%
%  Resume training routine, training set in "resume.mat"
%  Last column of the matrix are the labels
%

%
% Initializations
%
clear ; close all; %clc
trainMatrix = "resume500.mat";
testMatrix = 'test482.mat';
lambda = 1; 
num_labels = 2;
hidden_layer_size = 25000;	% change this later to equation based value
iterations = 1000;

fprintf ('Loading the training set %s\n',trainMatrix);
load (trainMatrix);
input_layer_size = size(X,2); % input layer is equal to the number of features (Columns)
% load "trainingset.mat";

fprintf('\nTraining Neural Network... \n')
fprintf('Input neurons: %d, Hidden neurons: %d, Iterations:%d\n', input_layer_size, hidden_layer_size,iterations);


%
% Create initial theta vectors and populate with random numbers and add the additional column
%
% fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
%
% Unroll the parameters for fminunc - an unconstrained function optimizer - requires
% the initial Thetas to be in a single vector representation
%
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

Theta1size = size (initial_Theta1);
Theta2size = size (initial_Theta2);

% Create a function handle (costFunction) to CostFunction
% Specifically costFunction is the handle, p is the function input arguments. So 
% this creates a new function "costFunction" that accepts a single argument. 
% The function also inherits the local environment.
costFunction = @(p) CostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters as one long vector)


% set the options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', iterations);

% *********************** using fminunc *****************************
%     FCN (CostFunction) should accept a vector (array) defining the unknown variables,
%
%     and return the objective function value, optionally with gradient.
%     'fminunc' attempts to determine a vector X such that 'FCN (X)' is a
%     local minimum.

[nn_params, cost] = fminunc(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));



fprintf(' \nWriting out the trained parameters Theta1 and Theta2 in trained.mat\n');
save ("trained.mat" ,"Theta1", "Theta2"); 


%  Run a prediction pass on the testMatrix

fprintf('Loading testing matrix %s\n', testMatrix);
load (testMatrix); % Load test matrix, X and y in memory.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


