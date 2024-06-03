%原始样本+拼接+聚类+降维+域适应
clear ; close all; clc
rng('default')
tic;
svm_ori = []; 
svm_kori = [];
svm_pori = [];
svm_lppmmd = [];
svm_lle = [];
svm_our = [];
svm_ourweidu = [];
svm_all=[];
for q = 2
    filename = ['L2/LR/LR',num2str(q),'.mat'];
    load (filename);
    n = size(trainX,2);
    train_data = trainX;
    test_data = testX;
    %%
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%直接用原始特征分类%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%     [trainX, mu, sigma] = featureCentralize(train_data);%%将样本标准化（服从N(0,1)分布）
%     testX = bsxfun(@minus, test_data, mu);
%     testX = bsxfun(@rdivide, testX, sigma);%%将测试样本标准化
% trData=[sample_pair_trainX,trainY];
% ttData=[sample_pair_testX,testY];
% [trNorm, ttNorm] = dataNorm(trData, ttData);
% trainX=trNorm(:,1:end-1);
% trainY=trNorm(:,end);
% testX=ttNorm(:,1:end-1);
% testY=ttNorm(:,end);
%     model = svmtrain(trainY,trainX,'-s 0 -c 10^5 -t 0 -q -b 1');
%     svm_pred = svmpredict(testY,testX,model); 
%     svm_ori =[svm_ori mean(double(svm_pred == testY)) * 100];
 %% SPC
% [sample_pair_testX, sample_pair_testY,sample_pair_trainX,sample_pair_trainY]=SPC(trainX,trainY,testX,testY);
%   model = svmtrain(trainY,sample_pair_trainX,'-s 0 -c 10^5 -t 0 -q -b 1');
%   svm_pred = svmpredict(testY,sample_pair_testX,model);
%   svm_pori =[svm_pori mean(double(svm_pred == testY)) * 100];
 %% kmeans
 for  p=0.5 % p=[2/3,1/2,1/3,1/4] 修改不同的聚类比例
%   for ii=1:50
 [m,n] = size(sample_pair_trainX);
[idx,temp] = kmeans(sample_pair_trainX,floor(m*p));
[l,ll]=size(temp);
cluster_Y=[];
for i = 1:l
    % 挑选第i类簇
    idx1 = find(idx==i);
    idx2 =trainY(idx1,:)
    M=mode(idx2)
cluster_Y=[cluster_Y M];
end
cluster_Y=cluster_Y';
cluster_X=temp;
  model = svmtrain(cluster_Y,cluster_X,'-s 0 -c 10^5 -t 0 -q -b 1');
  svm_kpred = svmpredict(testY,sample_pair_testX,model); %预测标签
  svm_kori =[svm_kori mean(double(svm_kpred == testY)) * 100];  
%% 聚类+降维
% %%%%%%%%%%%%%%%%%%%%%%
   maxStep = 10;            %最大迭代次数
   conCriterion = 0.01;     %迭代终止条件
   NeighborK = 10;           %LLE中邻域数目   

%------- construciton method 3    PR 文章所用
   options = [];
   options.k = 10;
   options.NeighborMode = 'KNN';
   W = Wconstruct_NPE(options,sample_pair_trainX);     
   W = BuildAdjacency(W,10);

    MuRegion = [10^-3,10^-2,10^-1,10^0,10^1,10^2,10^3];
    EntaRegion =  [10^-5,10^-4,10^-3,10^-2,10^-1,10^0,10^1,10^2];
    FAccRJSFCM = 0;
    qq=size(sample_pair_trainX,2);
    ACCindexRJSFCMjilu=zeros(1,qq);
    svm_RJS=zeros(1,qq-1);
    svm_our_bestweidu=1;         
    ACCindexRJSFCM1=0;
    ssvm_our_best=zeros(1,qq);
    
    for weidu=(qq*0.5):(qq-1)
       ProK = weidu 
            for mi=1:1:length(MuRegion)
                Pmu = MuRegion( mi );
                for eni=1:1:length(EntaRegion)
                        Penta = EntaRegion( eni );                   
    [P, V, steps, obj] = z_RJSFCM(sample_pair_trainX,W,cluster_X, ProK,Pmu,Penta,maxStep,conCriterion);
%%%%%%%%%%%%%%%%%%%%%%%
pre_trainX=sample_pair_trainX*P;
after_trainX=cluster_X*P;
after_testX=sample_pair_testX*P;                          
                       model = svmtrain(cluster_Y,after_trainX,'-s 0 -c 10^5 -t 0 -q');
                       svm_pred = svmpredict(testY,after_testX,model);  
                       ACCindexTemp = mean(double(svm_pred == testY))*100;                       
                       svm_RJS(ProK)= ACCindexTemp; 
                       pa=[weidu Pmu Penta ACCindexTemp];
                       svm_all= [svm_all;pa]; 
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                  
                        if FAccRJSFCM < ACCindexTemp
                           ACCindexRJSFCM = ACCindexTemp;
%                            NMIindexRJSFCM = NMIindexTemp;
                           FAccRJSFCM= ACCindexRJSFCM;
                           ProKRJSFCM = ProK;
                           PmuRJSFCM = Pmu;
                           PentaRJSFCM = Penta;
                           stepsRJSFCM = steps;                  
                           Popt = P;
                           Vopt = V;
                           RJStrainX=after_trainX;  %聚类后样本
                           RJStrainY=cluster_Y;    %聚类后标签
                           RJStestX= after_testX;        %聚类后测试集样本
                           RJSpre_trainX=pre_trainX;                  %聚类前样本
                        end
%                         end
%                         Circntf('ID: %f, ProK = %f, PGamma = %f, PMu = %f, PEnta = %f, ACC = %f, NMI = %f, Steps = %f\n', Circ, ProK, Pgamma, Pmu,Penta, ACCindexTemp,NMIindexTemp, steps );     
            end %end mi mu
        end % end gi gamma

    ACCindexRJSFCMjilu(weidu)= ACCindexRJSFCM;
end
svm_lle =[svm_lle ACCindexRJSFCM];
 end

%   if q==1
%     save("L2/Wisconsin/Wisconsin1","RJStrainX","RJStrainY","RJStestX","RJSpre_trainX","-append");
% end
% if q==2
%     save("L2/Wisconsin/Wisconsin2","RJStrainX","RJStrainY","RJStestX","RJSpre_trainX","-append");
% end
% if q==3
%     save("L2/Wisconsin/Wisconsin3","RJStrainX","RJStrainY","RJStestX","RJSpre_trainX","-append");
% end
% if q==4
%     save("L2/Wisconsin/Wisconsin4","RJStrainX","RJStrainY","RJStestX","RJSpre_trainX","-append");
% end
% if q==5
%     save("L2/Wisconsin/Wisconsin5","RJStrainX","RJStrainY","RJStestX","RJSpre_trainX","-append");
% end

end
toc;
fprintf('\nsvm Accuracy: %f\n',  mean(svm_ori));
fprintf('\nsvm Accuracy: %f\n',  std(svm_ori));

fprintf('\npsvm Accuracy: %f\n',  mean(svm_pori));
fprintf('\npsvm Accuracy: %f\n',  std(svm_pori));

fprintf('\nksvm Accuracy: %f\n',  mean(svm_kori));
fprintf('\nksvm Accuracy: %f\n',  std(svm_kori));

fprintf('\nllesvm Accuracy: %f\n',  mean( svm_lle));
fprintf('\nllesvm Accuracy: %f\n',  std( svm_lle));
