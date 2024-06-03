function  [acc,f1,AUc,ap,trainX_deep_best,testX_deep_best]=z_ESAE(trainX,testX,trainY,testY,type_num)
rng('default')%�����������,�����ʼȨֵ�������ʼ���ģ�����ÿ�ε�ʵ������һ��
f1=[ ];
AUc=[ ];
ap=[ ];
acc=[];
time=[];
tic;
T = constructT(trainY,type_num);               %��ǩ�����one-hot��ʽ��softmax���õ�
trainX_map  = mapminmax(trainX',0,1); %������һ��,�������ļ������sigmoid�����Χ��0-1
testX_map = mapminmax(testX',0,1);
record = [];
[m,n]= size(trainX);
[m1,n1]= size(testX);
best_accuracy = 1;

L2WeightRegularization = 0.1; %������ͷ�ϵ��
SparsityRegularization =4 ;     %ϵ����ͷ�ϵ��
SparsityProportion = 0.05;      %ϡ����� ����������������Ҫ����,�������Ա����������в���������ʱ���ã�
%%
for L2WeightRegularization=[0.0001,0.01,0.1]
%     for SparsityRegularization=[2,4,6,8,10]
%         for SparsityProportion =[0.01,0.03,0.05,0.07,0.09]
% L2WeightRegularization = 0.0001; %������ͷ�ϵ��

%%%%i,j,k ����������Ԫ����û�мȶ�������׼�򣬸�����������������������ȷ����ΧѰ��%%%
for i=40:20:100																																																															
    for j=10:10:50
        for k=5:5:25
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
        end
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

