clc;close all;clear;
rng('default')
ACC0_ALL = []; %ԭʼ�ռ�
ACC1_ALL = []; %���������ռ�1 
vote_all = []; %ͶƱ���
weight_all = [];%��Ȩ���
svm_pre = [];
svm_rec = [];
svm_f1 = [];
svm_all=[];
svm_lle=[];
svm_lppmmd=[];
time=[];
f1=[];
AUc=[];
ap=[];
for n =1:5
    tic;
    filename1 = ['dataset/AD/AD',num2str(n),'.mat'];
    filename2 = ['dataset/AD/AD',num2str(n),'_newDF.mat'];
    load(filename1);
    load(filename2);
%spc
[trainX, mu, sigma] = featureCentralize(trainX);%%��������׼��������N(0,1)�ֲ���
testX = bsxfun(@minus, testX, mu);
testX = bsxfun(@rdivide, testX, sigma);%%������������׼��
ALLdata = [trainX trainY];
[m,n]=size(ALLdata);
label=ALLdata(:,n);
%%train ѵ������ѵ������Ѱ�ҽ���
data=ALLdata(:,1:n-1);
k=2;
[new_data,new_label]=KN_datacreat_train(data,label,k);

sample_pair_trainX=[data,new_data];
sample_pair_trainY=[new_label];
%%test ���Լ���ѵ������Ѱ�ҽ���
k=1;
[new_data]=KN_datacreat_test(trainX,trainY,testX,testY,k);

sample_pair_testX=[testX,new_data];
sample_pair_testY=[testY];
%�������Ͻ�ά
 for  p=0.5 % p=[2/3,1/2,1/3,1/4] �޸Ĳ�ͬ�ľ������
%   for ii=1:50
 [m,n] = size(sample_pair_trainX);
[idx,temp] = kmeans(sample_pair_trainX,floor(m*p));
[l,ll]=size(temp);
cluster_Y=[];
for i = 1:l
    % ��ѡ��i���
    idx1 = find(idx==i);
    idx2 =trainY(idx1,:)
    M=mode(idx2)
cluster_Y=[cluster_Y M];
end
cluster_Y=cluster_Y';
cluster_X=temp;

%% ����+��ά
% %%%%%%%%%%%%%%%%%%%%%%
   maxStep = 10;            %����������
   conCriterion = 0.01;     %������ֹ����
   NeighborK = 10;           %LLE��������Ŀ   

%------- construciton method 3    PR ��������
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
    
    for weidu=(qq*0.5)
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
                           RJStrainX=after_trainX;  %���������
                           RJStrainY=cluster_Y;    %������ǩ
                           RJStestX= after_testX;        %�������Լ�����
                           RJSpre_trainX=pre_trainX;                  %����ǰ����
                        end
%                         end
%                         Circntf('ID: %f, ProK = %f, PGamma = %f, PMu = %f, PEnta = %f, ACC = %f, NMI = %f, Steps = %f\n', Circ, ProK, Pgamma, Pmu,Penta, ACCindexTemp,NMIindexTemp, steps );     
            end %end mi mu
        end % end gi gamma

    ACCindexRJSFCMjilu(weidu)= ACCindexRJSFCM;
end
svm_lle =[svm_lle ACCindexRJSFCM];
 end
%����Ӧ
  Xt_train = RJSpre_trainX'; %%����ǰ�����ݼ��ͱ�ǩ; 
  target_label_train = trainY';
  Xs_train =RJStrainX'; %%���������ݼ��ͱ�ǩ 
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
 for l=5  % ����ά��
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
  Yat=[source_label_train]';%target_label_train ��ǩ
  Xat=[X_train(:,1:length(source_label_train))]';   
  Xt_test_new=Xt_test_new';
  
    [Xat, mu, sigma] = featureCentralize(Xat);%%��������׼��������N(0,1)�ֲ���
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
   svm_pred = svmpredict(testY,Xt_test_new,model); %Ԥ���ǩ
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
%encode
 %[trainX_deep_cluster0,testX_deep_cluster0,trainX_deep_cluster1,testX_deep_cluster1]=ESAE(trainX,trainY,testX,testY,trainX1,trainY1,testX1);
  
%classification
    train_data=trainX_deep_cluster0;
    test_data=testX_deep_cluster0;        
    train_data1=trainX_deep_cluster1;
    test_data1=testX_deep_cluster1;
    %�����ռ�0
    [trainX, mu, sigma] = featureCentralize(train_data);%%��������׼��������N(0,1)�ֲ���
    testX = bsxfun(@minus, test_data, mu);
    testX = bsxfun(@rdivide, testX, sigma);%%������������׼��
    model = svmtrain(trainY,trainX,'-s 0 -c 10^5 -t 0 -q -b 1');
   [predictLable0, ~, scores0] = svmpredict(testY,testX, model,'-b 1');
    acc0 =mean(double(predictLable0 == testY)) * 100;

    %�����ռ�1
%      [acc1,predictLable1,scores1] = predict(trainX_deep_cluster1,trainY_ICM,testX_deep_cluster1,testY,type_num);
%     model = svmtrain(trainY_ICM,trainX_deep_cluster1,'-s 0 -c 10^5 -t 0 -q');
%     predictLable1 = svmpredict(testY,testX_deep_cluster1,model); %Ԥ���ǩ
    [trainX, mu, sigma] = featureCentralize(train_data1);%%��������׼��������N(0,1)�ֲ���
    testX = bsxfun(@minus, test_data1, mu);
    testX = bsxfun(@rdivide, testX, sigma);%%������������׼��
    model = svmtrain(RJStrainY,trainX,'-s 0 -c 10^5 -t 0 -q -b 1');
   [predictLable1, ~, scores1] = svmpredict(testY,testX, model,'-b 1');
    acc1 =mean(double(predictLable1 == testY)) * 100;
%    %%
    % %%����ͶƱ�� %%%%%   
    m = size(testY,1);%����������
    K = zeros(m,1);
    for i = 1:m
        sumw = [];
        for j = 1:type_num
            vote = 0; 
            if predictLable0(i)==j
                vote = vote + 1;
            end 
            if predictLable1(i)==j
                vote = vote + 1;
            end 
%             if predictLable2(i)==j
%                 vote = vote + 1;
%             end    
            sumw = [sumw vote];
        end
        [value,ind] =max(sumw);
        K(i) = ind;      
    end   
    vote_acc =  mean(double(K == testY)) * 100; 

  %  save("CM/CMpendigits","testY","K");%������Ա�ǩtestY��Ԥ���ǩK

  
  %%%��Ȩ����%%%%%
    acc_best = 1;
    N = round(size(testX,1)/2);
    validscores0 = scores0(1:N,:);
    testscores0 = scores0(N+1:end,:);
    validscores1 =scores1(1:N,:);
    testscores1 = scores1(N+1:end,:);
%     validscores2 = scores2(1:N,:);
%     testscores2 = scores2(N+1:end,:);
  %%
    validLable0 = predictLable0(1:N);
    testLable0 = predictLable0(N+1:end);
    validLable1 = predictLable1(1:N);
    testLable1 = predictLable1(N+1:end);
  %%
    for a=0:0.1:1
        b=1-a;
        validLable = round(a*validLable0 + b*validLable1);
        acc = mean(validLable == testY(1:N))*100;
        if acc > acc_best
            x=a;
            y=b;  
            acc_best = acc;
        end
    end
    validLable = round(x*validLable0 + y*validLable1);%��֤��
    testLable =  round(x*testLable0 + y*testLable1);
    weight_acc = mean(testLable == testY(N+1:end))*100;

    ACC0_ALL = [ACC0_ALL;acc0];
    ACC1_ALL = [ACC1_ALL;acc1];
    vote_all = [vote_all;vote_acc];
    weight_all = [weight_all; weight_acc];
toc; 
time=[time toc];
end
fprintf('\nԭʼ�����ռ� Accuracy: %f\n', mean(ACC0_ALL));
fprintf('\nԭʼ ����: %f\n', std(ACC0_ALL));
fprintf('\nһ�������ռ� Accuracy: %f\n', mean(ACC1_ALL));
fprintf('\nһ�� ����: %f\n', std(ACC1_ALL));
fprintf('\nvote Accuracy: %f\n', mean(vote_all));
fprintf('\nvote ����: %f\n', std(vote_all));
fprintf('\nweight Accuracy: %f\n', mean(weight_all));
fprintf('\nweight ����: %f\n', std(weight_all));
fprintf('\nsvm time: %f\n',  mean(time))
fprintf('\nsvm time����: %f\n',  std(time));