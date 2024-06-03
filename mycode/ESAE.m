clear ; close all; clc
rng('default')%�����������,�����ʼȨֵ�������ʼ���ģ�����ÿ�ε�ʵ������һ��
f1=[ ];
AUc=[ ];
ap=[ ];
acc=[];
time=[];
for q=1:5
    tic;
    filename1 = ['L2/pendigits/pendigits',num2str(q),'.mat'];
    load (filename1);
% [trainX1,trainY1] = k_means(trainX,train,type_num,0.5); % �ھ��������ռ���ѵ��������
% [trainX2,trainY2] = k_means(trainX1,trainY1,type_num,0.5); % �ھ��������ռ���ѵ��������
% trainX = trainX1;
% trainY = trainY1;
% testX = testX1;
% trainX = RJStrainX;
ALLdata = [trainX trainY];
[m,n]=size(ALLdata);
label=ALLdata(:,n);
%%train ѵ������ѵ������Ѱ�ҽ���
data=ALLdata(:,1:n-1);
k=2;
[new_data,new_label]=KN_datacreat_train(data,label,k);

trainX1=[data,new_data];
trainY1=[new_label];
%%test ���Լ���ѵ������Ѱ�ҽ���
k=1;
[new_data]=KN_datacreat_test(trainX,trainY,testX,testY,k);

testX1=[testX,new_data];
testY=[testY];
trainX = trainX1;
testX = testX1;
[trainX,trainY] = k_means(trainX,trainY,type_num,0.5); % �ھ��������ռ���ѵ��������
% trainY = trainY1;
% testX = RJStestX;
% trainX = trainX2;
% trainY = trainY2;
% testX = testX2;
% testY = testY2;
T = constructT(trainY,type_num);               %��ǩ�����one-hot��ʽ��softmax���õ�
trainX_map  = mapminmax(trainX',0,1); %������һ��,�������ļ������sigmoid�����Χ��0-1
testX_map = mapminmax(testX',0,1);
record = [];
[m,n]= size(trainX);
[m1,n1]= size(testX);
best_accuracy = 1;

L2WeightRegularization = 0.001; %������ͷ�ϵ��
SparsityRegularization =4 ;     %ϵ����ͷ�ϵ��
SparsityProportion = 0.05;      %ϡ����� ����������������Ҫ����,�������Ա����������в���������ʱ���ã�
%%
% for L2WeightRegularization=[0.0001]
%     for SparsityRegularization=2
%         for SparsityProportion = [0.05]
% L2WeightRegularization = 0.0001; %������ͷ�ϵ��

%%%%i,j,k ����������Ԫ����û�мȶ�������׼�򣬸�����������������������ȷ����ΧѰ��%%%
for i = 100:20:120																																																																
    for j = 50:10:60
        for k = 15:5:20
            hiddenSize = i;
            autoenc1 = trainAutoencoder(trainX',hiddenSize,...%�������������ݸ�ʽ: d*N,�ú���Ĭ�϶��������ݹ�һ������
                'MaxEpochs',1000,...  %��������
                'L2WeightRegularization',L2WeightRegularization,...%��ʧ������L2Ȩ�ص�������ϵ����=0.0001
                'SparsityRegularization',SparsityRegularization,...%����ϡ���������Գɱ�������Ӱ���ϵ����=4
                'SparsityProportion',SparsityProportion,...%���������ز������ϡ����
                'encoderTransferFunction','logsig',...%�����������
                'DecoderTransferFunction','logsig');  %�����������
            %Extract the features in the hidden layer.
            features1 = encode(autoenc1,trainX');%ʹ���Ա��������������ݽ��б���
            %Ƕ��ԭʼ��������һ�����������������
            features1 = [features1;trainX_map];
            features1 = featureChoose(features1, i); 

            hiddenSize = j;
            autoenc2 = trainAutoencoder(features1,hiddenSize,...
                 'MaxEpochs',1000,...
                'L2WeightRegularization',L2WeightRegularization,...
                'SparsityRegularization',SparsityRegularization,...
                'SparsityProportion',SparsityProportion,...
                'encoderTransferFunction','logsig',...
                'DecoderTransferFunction','logsig');
            %Extract the features in the hidden layer.
%ʹ�õ�һ���Ա������������Ϊ�ڶ����Ա����������롣
            features2 = encode(autoenc2,features1);
            features2 = [features2;trainX_map];
            features2 = featureChoose(features2, j); 

            hiddenSize = k;
            autoenc3 = trainAutoencoder(features2,hiddenSize,...
                 'MaxEpochs',1000,...
                'L2WeightRegularization',L2WeightRegularization,...
                'SparsityRegularization',SparsityRegularization,...
                'SparsityProportion',SparsityProportion,...
                'encoderTransferFunction','logsig',...
                'DecoderTransferFunction','logsig');
            %Extract the features in the hidden layer.
            features3 = encode(autoenc3,features2);
           %%%%������Ԥѵ���׶�%%%%%
            
            
            %����softmax����㣬ʹ��ѵ�����ݵı�ǩ���мල��ʽѵ�� softmax �㡣
            softnet = trainSoftmaxLayer(features3,T,'LossFunction','crossentropy'); 
            %trainSoftmaxLayer(���������������ǩ������(ѭ������)������ֵ)

           
            
            deepnet = stack(autoenc1,autoenc2,autoenc3,softnet);%�����ջ���������Ա������еı�������������ȡ�����������Խ��Ա������еı������� softmax ��ѵ���һ�����γ����ڷ���Ķѵ�����
            
            deepnet = train(deepnet,trainX',T);%����΢����ͨ�����мල��ʽ����ѵ����������ѵ��������΢������
            
            %��ѵ���õ������ѵ�����Ͳ��Լ����б���
            train_deepFeature = coding(deepnet,m, trainX_map');%ѵ���������������ȡ
            test_deepFeature = coding(deepnet,m1, testX_map');%��������������ȡ

%             model = svmtrain(trainY,train_deepFeature,'-s 0 -c 10^5 -t 0 -q'); %ѵ��������
%             svm_pred = svmpredict(testY,test_deepFeature,model); 
            model = svmtrain(trainY,train_deepFeature,'-s 0 -c 10^5 -t 0 -q -b 1');
            [svm_pred, ~, prob_estimates] = svmpredict(testY,test_deepFeature, model,'-b 1');
            accuracy = mean(double(svm_pred == testY)) * 100; 
            if(accuracy > best_accuracy)
                trainX_deep_best = train_deepFeature;
                testX_deep_best = test_deepFeature;
                best_accuracy = accuracy;
                svm_pred_best=svm_pred;
                prob_estimates_best=prob_estimates;
                network = deepnet;
%                 BestL2=L2WeightRegularization;
%                 BestSR=SparsityRegularization;
%                 BestSP=SparsityProportion;
%                 besti=i
%                 bestj=j
%                 bestk=k
                        end
    end
end
            end
%         end
%     end 
% end
toc;
NN=size(testX,1);
L=constructT(testY,type_num)';
Y=constructT(svm_pred_best,type_num)';
%���ʾ���
P=fliplr(prob_estimates_best);
toc;
[F1] = f1score(type_num,svm_pred_best,testY);
[auc] = AUC( L,Y,NN,type_num);
[AP,PR,SPR,Precision,Recall]=Ap(P,L,Y,NN,type_num);
f1=[f1 F1];
AUc=[AUc auc];
ap=[ap AP];
acc=[acc best_accuracy];
time=[time toc];
end
% %%��¼��������ֵ
% besti
% bestj
% bestk
%   BestL2
%    BestSR
%    BestSP

%  view(deepnet)
view(network)
network1 = network;
IMC_trainX_deep = trainX_deep_best;%һ�ξ��������ռ�����ȡ����������
IMC_testX_deep = testX_deep_best;
% network2 = network;
% trainX_deep_cluster2 = trainX_deep_best;%���ξ��������ռ�����ȡ����������
% testX_deep_cluster2 = testX_deep_best;
% save("L0/AD/AD3_newDF","IMC_trainX_deep","IMC_testX_deep","network1","trainY","testY","-append");%����������ɵ�����
% save("sjcy/AD/AD_1_newDF","trainX_deep_best","testX_deep_best","network","-append"); %�����������
 fprintf('\nsvm acc: %f\n',  mean(acc));
fprintf('\nsvm acc����: %f\n',  std(acc));
fprintf('\nsvm f1: %f\n',  mean(f1));
fprintf('\nsvm f1����: %f\n',  std(f1));
fprintf('\nsvm AUC: %f\n',  mean(AUc));
fprintf('\nsvm AUC����: %f\n',  std(AUc));
fprintf('\nsvm ap: %f\n',  mean(ap));
fprintf('\nsvm ap����: %f\n',  std(ap));
fprintf('\nsvm time: %f\n',  mean(time));
fprintf('\nsvm time����: %f\n',  std(time));

 
