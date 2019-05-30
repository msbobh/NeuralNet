function W = randInitializeWeights(L_in, L_out)
% Create a random iintialziation for a neural network of arbitrary layers and hidden units
% randInitializeWeights(L_in, L_out)
%    L_in = the number incoming connections
%    L_out = the number of  outgoing connections
%    
%
%   W returns a matrix of size(L_out, 1 + L_in) a 
%   column of one's is added to the first col of W for the bias terms
%

% Create empty return matrix 
W = zeros(L_out, 1 + L_in);

% 
% Initialize W randomly with small values close to zero, this will break 
% symmetry while training the neural network.


epsilon_init = 0.12;
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;

end
