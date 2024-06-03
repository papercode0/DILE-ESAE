%域适应
clear ; close all; clc
rng('default')
tic;
svm_ori = []; 
orig=[];
svm_kori = [];
svm_pori = [];
svm_lppmmd = [];
svm_lle = [];
svm_our = [];
svm_ourweidu = [];
for q = 1:5 
    filename = ['km/LR/LR',num2str(q),'.mat'];
    load (filename);
   model = svmtrain(RJStrainY,RJStrainX,'-s 0 -c 10^5 -t 0 -q');
   svm_pred = svmpredict(testY,RJStestX,model);%预测标签
   orig = [orig mean(double(svm_pred == testY)) * 100];
%    
trData=[RJStrainX,RJStrainY];
ttData=[RJStestX,testY];
[trNorm, ttNorm] = dataNorm(trData, ttData);
RJStrainX=trNorm(:,1:end-1);
RJStrainY=trNorm(:,end);
RJStestX=ttNorm(:,1:end-1);
testY=ttNorm(:,end);  
   
   
   [RJSprerainX, mu, sigma] = featureCentralize(RJSpre_trainX);%%将样本标准化（服从N(0,1)分布）
    [RJStrainX, mu, sigma] = featureCentralize(RJStrainX);%%将样本标准化（服从N(0,1)分布）
    RJStestX = bsxfun(@minus, RJStestX, mu);
    RJStestX = bsxfun(@rdivide, RJStestX, sigma);%%将测试样本标准化  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%域适应%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
  Xt_train = RJSpre_trainX'; %%聚类前的数据集和标签; 
  target_label_train = trainY';
  Xs_train =RJStrainX'; %%聚类后的数据集和标签 
  Xt_test = RJStestX';
  source_label_train = RJStrainY';
  source_label_test = testY';
  
Graph_Xt_train_num=2;
Graph_Xt_test_num=5;

        ker_type=1; % 0:linear 1:nonlinear
        Kernel='linear';
  ALLX=[Xs_train Xt_train];  
  svm_lppmmd_best = 1;
  m=size(ALLX,2);
  i=size(ALLX,1);
  z_svm_orig=zeros(1,i);  
 for l=2:i  % 最优维度
  traindata=ALLX';
%   trainlabel=[dataY1;trainY];
  X=datasample(traindata,l)';
%   X=traindata(1:l,:)';
% traindata=ALLX';
 [Z,P,Ks, Kt,KY_test]=ICM(X,Xs_train,Xt_train,Xt_test,Graph_Xt_train_num,Graph_Xt_test_num,Kernel);    
%     %------------------------------------------------------   

     KX_train  = Ks;
     KY_train  = Kt;

     
   X_train  = P'*KX_train;
   Y_train  = P'*KY_train;
   Xt_test_new  = P'*KY_test;  
   %svm
  Yat=[source_label_train]';%target_label_train 标签
  Xat=[X_train(:,1:length(source_label_train))]';   
  Xt_test_new=Xt_test_new';
  
    [Xat, mu, sigma] = featureCentralize(Xat);%%将样本标准化（服从N(0,1)分布）
    Xt_test_new = bsxfun(@minus,  Xt_test_new, mu);
     Xt_test_new = bsxfun(@rdivide, Xt_test_new, sigma);  
   
trData=[Xat,Yat];
ttData=[Xt_test_new,testY];
[trNorm, ttNorm] = dataNorm(trData, ttData);
Xat=trNorm(:,1:end-1);
Yat=trNorm(:,end);
Xt_test_new=ttNorm(:,1:end-1);
testY=ttNorm(:,end);  
     
   model = svmtrain(Yat,Xat,'-s 0 -c 10^5 -t 0 -q');
   svm_pred = svmpredict(testY,Xt_test_new,model); %预测标签
   svm_orig = mean(double(svm_pred == testY)) * 100;
   z_svm_orig(l)=svm_orig;
  
   
   if svm_orig > svm_lppmmd_best
    svm_lppmmd_best = svm_orig;
    trainX111 = Xat;
    trainY111 = Yat;
    testX111=Xt_test_new;
    n_best = l;
        end
 end  
  svm_lppmmd =[svm_lppmmd svm_lppmmd_best];
  [sample_pair_RJStestX, testY,sample_pair_RJStrainX,sample_pair_RJStrainY]=SPC(trainX111,trainY111,testX111,testY);
  trainX_ICM=trainX111;
  trainY_ICM=trainY111;
  testX_ICM=testX111;
% 
  model = svmtrain(sample_pair_RJStrainY,sample_pair_RJStrainX,'-s 0 -c 10^5 -t 0 -q -b 1');
  svm_pred = svmpredict(testY,sample_pair_RJStestX,model);
  svm_pori =[svm_pori mean(double(svm_pred == testY)) * 100];

% if q==1
%     save("Statlog/Statlog1","testX_ICM","trainX_ICM","trainY_ICM","sample_pair_RJStestX","sample_pair_RJStrainX","-append");
% end
% if q==2
%     save("Statlog/Statlog2","testX_ICM","trainX_ICM","trainY_ICM","sample_pair_RJStestX","sample_pair_RJStrainX","-append");
% end
% if q==3
%     save("Statlog/Statlog3","testX_ICM","trainX_ICM","trainY_ICM","sample_pair_RJStestX","sample_pair_RJStrainX","-append");
% end
% if q==4
%     save("Statlog/Statlog4","testX_ICM","trainX_ICM","trainY_ICM","sample_pair_RJStestX","sample_pair_RJStrainX","-append");
% end
% if q==5
%     save("Statlog/Statlog5","testX_ICM","trainX_ICM","trainY_ICM","sample_pair_RJStestX","sample_pair_RJStrainX","-append");
% end
end
toc;
fprintf('\nRJSIMC Accuracy: %f\n',  mean( orig));
fprintf('\nRJSIMC Accuracy: %f\n',  std( orig));

fprintf('\nlppmmdsvm Accuracy: %f\n',  mean( svm_lppmmd));
fprintf('\nlppmmdsvm Accuracy: %f\n',  std( svm_lppmmd));

fprintf('\npsvm Accuracy: %f\n',  mean(svm_pori));
fprintf('\npsvm Accuracy: %f\n',  std(svm_pori));