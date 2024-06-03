生成样本对
rng('default')
clear ; close all; clc
% ALLdata= importdata('d2/AD.mat');
tic;
svm_spc=[];SVM_SPC=[];
for q=1:5
    
filename = ['L2/LR/LR',num2str(q),'.mat'];
load (filename);

    [trainX, mu, sigma] = featureCentralize(trainX);%%将样本标准化（服从N(0,1)分布）
    testX = bsxfun(@minus, testX, mu);
    testX = bsxfun(@rdivide, testX, sigma);%%将测试样本标准化
% trData=[trainX,trainY];
% ttData=[testX,testY];
% [trNorm, ttNorm] = dataNorm(trData, ttData);
% trainX=trNorm(:,1:end-1);
% trainY=trNorm(:,end);
% testX=ttNorm(:,1:end-1);
% testY=ttNorm(:,end);
% model = svmtrain(trainY,trainX,'-s 0 -c 10^5 -t 0 -q -b 1');
% svm_pred = svmpredict(testY,testX,model); %预测标签
% svm_ori =mean(double(svm_pred == testY)) * 100;
% SVM_SPC=[SVM_SPC svm_ori]
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
[new_data]=KN_datacreat_test(trainX,trainY,testX,testY,k);

sample_pair_testX=[testX,new_data];
sample_pair_testY=[testY];

% sample_pair_RJStestX1=sample_pair_testX;
% sample_pair_RJStrainX1=sample_pair_trainX;
%%
%     [sample_pair_RJStrainX, mu, sigma] = featureCentralize(sample_pair_RJStrainX);%%将样本标准化（服从N(0,1)分布）
%     sample_pair_RJStestX = bsxfun(@minus, sample_pair_RJStestX, mu);
%     sample_pair_RJStestX = bsxfun(@rdivide, sample_pair_RJStestX, sigma);%%

% trData=[sample_pair_RJStrainX,trainY_ICM];
% ttData=[sample_pair_RJStestX,testY];
% [trNorm, ttNorm] = dataNorm(trData, ttData);
% sample_pair_RJStrainX=trNorm(:,1:end-1);
% trainY_ICM=trNorm(:,end);
% sample_pair_RJStestX=ttNorm(:,1:end-1);
% testY=ttNorm(:,end); 

 model = svmtrain(trainY,sample_pair_trainX,'-s 0 -c 10^5 -t 0 -q -b 1');
 svm_kpred = svmpredict(sample_pair_testY,sample_pair_testX,model); %预测标签
 svm_kori =mean(double(svm_kpred == testY)) * 100;
 svm_spc =[svm_spc svm_kori];

end


% save("PD5.mat","sample_pair_RJStrainX","sample_pair_RJStestX","-append");
toc;
fprintf('\nsvm Accuracy: %f\n',  mean(SVM_SPC));
fprintf('\nsvm Accuracy方差: %f\n',  std(SVM_SPC));
fprintf('\nsvm Accuracy: %f\n',  mean(svm_spc));
fprintf('\nsvm Accuracy方差: %f\n',  std(svm_spc));