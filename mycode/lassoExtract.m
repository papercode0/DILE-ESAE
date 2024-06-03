function [trainX_lasso,testX_lasso] = lassoExtract(trainX,trainY,testX)

if(size(trainX,1)<5)
    [B,FitInfo] = lasso(trainX,trainY,'Alpha',1,'CV',2);
    %�Ա���x�������y��CVʹ�ý�����֤��2�۽�����֤��Alpha����=0Ϊ��ع飬��=1Ϊlasso�ع飩��
    %����ֵΪ��B��Ȩ��ϵ����fitinfo:ģ����Ϣ��
else
    [B,FitInfo] = lasso(trainX,trainY,'Alpha',1,'CV',5);
end
% idxLambdaMinMSE = FitInfo.IndexMinMSE;%����С��������Ӧ��ģ�ͣ�
% coef = B(:,idxLambdaMinMSE);
% idx1SE = FitInfo.Index1SE;%����С��������Ӧ��ģ�ͣ�
% coef = B(:,idx1SE);
coef = B(:,5);%%����������������������������������
%Ѱ��ϵ���еķ����~=0Ϊ������0��
index = find(coef ~= 0);
t = size(index,1);
trainX_lasso = zeros(size(trainX,1),t);
testX_lasso = zeros(size(testX,1),t);
for i = 1:t
    trainX_lasso(:,i) = trainX(:,index(i));
    testX_lasso(:,i) = testX(:,index(i));
end
end

