function [sample_pair_testX, sample_pair_testY,sample_pair_trainX,sample_pair_trainY]=SPC(trainX,trainY,testX,testY)   
ALLdata = [trainX trainY];
[m,n]=size(ALLdata);
label=ALLdata(:,n);
%%train 训练集从训练集中寻找近邻
data=ALLdata(:,1:n-1);
k=2;
[new_data,new_label]=KN_datacreat_train(data,label,k);

sample_pair_trainX=[data,new_data];
sample_pair_trainY=[new_label];
%%test 测试集从训练集中寻找近邻
k=1;
[new_data]=KN_datacreat(trainX,trainY,testX,testY,k);

sample_pair_testX=[testX,new_data];
sample_pair_testY=[testY];
