function output = sigmoid_func(X)
% sigmoid function
% input parameters: x = m x n matrix, m is the block size, 
output=1./(1+exp(-X));
% n is the number of features
% output parameters: m x n matrix
% Your code here
% Your code end here
end