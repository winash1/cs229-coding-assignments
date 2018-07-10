function [dW1, db1, dW2, db2] = backward_prop(X, y_onehot, W1, b1, W2, b2, lambda)
% backward propagation for our 1 layer network
% input parameters
% X is our m x n dataset, where m = number of samples, n = number of
% features
% y is the length m label vector for each sample
% W1 is our h1 x n weight matrix, where h1 = number of hidden units in
% layer 1
% b1 is the length h1 column vector of bias terms associated with layer 1
% W2 is the c x h1 weight matrix, where c = number of classes
% b2 is the length h2 column vector of bias terms associated with the output
% output parameters
% returns the gradient of W1, b1, W2, b2 as dW1, db1, dW2, db2
% Your code here
[h_output, prob, loss] = forward_prop(X, y_onehot, W1, b1, W2, b2, lambda);
dW2=(y_onehot-prob)'*h_output';
db2=sum((y_onehot-prob)',2);
db1=sum(((y_onehot-prob)*W2)'.*(h_output.*(1-h_output)),2);
dW1=((y_onehot-prob)*W2)'.*(h_output.*(1-h_output))*X;
dW1=dW1./length(X(:,1));
dW2=dW2./length(X(:,1));
db2=db2/length(X(:,1));
db1=db1/length(X(:,1));