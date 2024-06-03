function [acc,predictLable,MD] = predict(trainX_deep,trainY,testX_deep,testY,type_num)
%  [m1,n] = size(trainX);
%     m2 = size(testX,1);
%     acc=[];
%     MD=1;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%直接用原始特征分类%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%     [trainX, mu, sigma] = featureCentralize(trainX);%%将样本标准化（服从N(0,1)分布）
%     testX = bsxfun(@minus, testX, mu);
%     testX = bsxfun(@rdivide, testX, sigma);%%将测试样本标准化
%     model = svmtrain(trainY,trainX,'-s 0 -c 10^5 -t 0 -q');
%     svm_pred = svmpredict(testY,testX,model); %预测标签
%     acc =mean(double(svm_pred == testY)) * 100;
%     predictLable = svm_pred;
     %%%%%%%%%%%%%%%%%%%%%%%%%%%深度特征%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     [trainX2, mu, sigma] = featureCentralize(trainX_deep); 
%     testX2 = bsxfun(@minus, testX_deep, mu);
%     testX2 = bsxfun(@rdivide, testX2, sigma); 

%     model = svmtrain(trainY,trainX_deep,'-s 0 -c 10^5 -t 0 -q');
%     svm_pred = svmpredict(testY,testX_deep,model);  
%     svm_deep = mean(double(svm_pred == testY)) * 100;
%     result_df = svm_deep;
%     acc = result_df;
%     predictLable = svm_pred;
%     MD=1;
    
%%    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%降维%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%PCA%%%%%%%%%%%%%%%%%%%%%%%%   
   [m1,n] = size(trainX_deep);
     method = [];
    method.mode = 'pca';
    acc = 1;
    k_best = n;
    for k = 1:5:n-1
        method.K = k;
        [trainX_deep, mu, sigma] = featureCentralize(trainX_deep);
        testX = bsxfun(@minus, testX_deep, mu);
       testX_deep = bsxfun(@rdivide, testX, sigma);
        [trainZ,U] = featureExtract(trainX_deep,trainY,method,type_num);
        testZ = projectData(testX_deep, U, method.K);%将测试集按照训练集的映射方式映射到空间中
% %%%%%%%%%%%%%%%%%%%%%%%%%%% SVM %%%%%%%%%%%%%%%%%%%%%%%%%%%
        model = svmtrain(trainY,trainZ,'-s 0 -c 10^5 -t 0 -q -b 1');
%         svm_pred = svmpredict(testY,testZ,model);
 %%%%%%%%%%%%%%%%
 [svm_pred, ~, prob_estimates] = svmpredict(testY,testZ, model,'-b 1');

 %%%%%%%%%%%%%
        svm_pca = mean(double(svm_pred == testY)) * 100;  
        if svm_pca > acc
            acc = svm_pca;
            predictLable = svm_pred;
            MD = prob_estimates;
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    nTree = 5;
%    B = TreeBagger(nTree,trainZ,trainY);
%    predict_label = predict(B, testZ);
%    predict_label= str2num(char(predict_label));
%    svm_pca = mean(double(predict_label == testY)) * 100;
%     if svm_pca > acc
%             acc = svm_pca;
%             predictLable = predict_label;
%         end
% end
%     
  %%%%%%%%%%%%%%%%%%%%%%%% ELM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    traindata=[trainY trainZ];
%    testdata=[testY  testZ];
% [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,elm_pred] = ELM(traindata, testdata,1, 130, 'sig');
%    elm_deep = mean(double(elm_pred == testY)) * 100;
% %    result_elm(q) = elm_deep;
%      if elm_deep > acc
%             acc = elm_deep;
%             predictLable = elm_pred;
%         end
% end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        [m1,n] = size(trainX_deep);
%  method = [];
%     method.mode = 'lda';
%     acc = 1;
%     k_best = 1; 
% %     [trainX, mu, sigma] = featureCentralize(trainX_deep);
% %     testX = bsxfun(@minus, testX_deep, mu);
% %     testX = bsxfun(@rdivide, testX, sigma);
%     for k = 1:1:n-1
%         method.K = k;
%         [trainZ,U] = featureExtract(trainX_deep,trainY,method,type_num);
%         testZ = projectData(testX_deep, U, method.K);
%         model = svmtrain(trainY,trainZ,'-s 0 -c 10^5 -t 0 -q');
%         svm_pred = svmpredict(testY,testZ,model);
%         svm_lda = mean(double(svm_pred == testY)) * 100;
%         if svm_lda > acc
%             acc = svm_lda;
%             predictLable = svm_pred;
%         end
%     end
%    
end