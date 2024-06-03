function [trainX_lasso,testX_lasso] = lassoExtract(trainX,trainY,testX)

if(size(trainX,1)<5)
    [B,FitInfo] = lasso(trainX,trainY,'Alpha',1,'CV',2);
    %自变量x，因变量y，CV使用交叉验证，2折交叉验证，Alpha，α=0为岭回归，α=1为lasso回归）；
    %返回值为（B：权重系数，fitinfo:模型信息）
else
    [B,FitInfo] = lasso(trainX,trainY,'Alpha',1,'CV',5);
end
% idxLambdaMinMSE = FitInfo.IndexMinMSE;%（最小均方误差对应的模型）
% coef = B(:,idxLambdaMinMSE);
% idx1SE = FitInfo.Index1SE;%（最小均方误差对应的模型）
% coef = B(:,idx1SE);
coef = B(:,5);%%？？？？？？？？？？？？？？？？？
%寻找系数中的非零项（~=0为不等于0）
index = find(coef ~= 0);
t = size(index,1);
trainX_lasso = zeros(size(trainX,1),t);
testX_lasso = zeros(size(testX,1),t);
for i = 1:t
    trainX_lasso(:,i) = trainX(:,index(i));
    testX_lasso(:,i) = testX(:,index(i));
end
end

