function [svm,K ] = relief(trainx,trainy,testx,testy)
%UNTITLED6 此处显示有关此函数的摘要
%   此处显示详细说明
 [ranks,weights] = relieff(trainx,trainy,5,'method','classification','categoricalx','on');

 for k = 1:5:150
     trainX = [];
     testX = [];
     for j = 1:k
         trainX = [trainX;trainx(:,ranks(j))];
         testX = [testX;testx(:,ranks(j))];
     end
     svm = 0;
     model = svmtrain(trainy,trainX,'-s 0 -c 10^5 -t 0 -q');
     svm_pred = svmpredict(testy,testX,model);
     svm1 = mean(double(svm_pred == testy)) * 100;
     if svm1 > svm
         svm = svm1;
         K = k;
     end   
end

