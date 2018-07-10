function prob = softmax_func(X)
% softmax function
% input parameters: x = m x c matrix, m is the block size, 
% n is the number of classes
% output parameters: : m x c matrix
% Your code here
c=1*ones(size(X));
prob=bsxfun(@rdivide, exp(X+c),sum(exp(X+c),1));
% Your code end here
end
