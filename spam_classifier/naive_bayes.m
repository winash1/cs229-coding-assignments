clear
[spmatrix, tokenlist, trainCategory] = readMatrix('MATRIX.TRAIN');
trainMatrix = full(spmatrix);
numTrainDocs = size(trainMatrix, 1);
numTokens = size(trainMatrix, 2);
train_y=full(trainCategory);
theta_kgiven1=(sum(diag(train_y)*trainMatrix)+1)./((sum(sum(diag(train_y)*trainMatrix,2)))+numTokens);
theta_kgiven0=(sum(diag(~train_y)*trainMatrix)+1)./((sum(sum(diag(~train_y)*trainMatrix,2)))+numTokens);
theta=sum(train_y)/numTrainDocs;
[spmatrix_test, tokenlist_test, category_test] = readMatrix('MATRIX.TEST');
testMatrix = full(spmatrix_test);
numTestDocs_test = size(testMatrix, 1);
numTokens_test = size(testMatrix, 2);
sum(testMatrix*diag(log((theta_kgiven0))),2)-sum(testMatrix*diag(log((theta_kgiven1))),2);
prob=1./(1+exp(sum(testMatrix*diag(log((theta_kgiven0))),2)+log((1-theta))-log(theta)-sum(testMatrix*diag(log((theta_kgiven1))),2)));
out=prob-0.499999999999999>0;
test_category=full(category_test);
error=1-(sum(~(xor(test_category',out)))/800);
[temp,origpos] = sort(log(theta_kgiven1)-log(theta_kgiven0),'descend');
p=origpos(1:10);
C = strsplit(tokenlist);
C(p)