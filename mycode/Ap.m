function [aP PR SPR Precision Recall]=Ap(prob_estimates,testY,svm_pred,m,type_num)
aP=[];
for i=1:type_num
prob_estimates1=prob_estimates(:,i);
testY1=testY(:,i);
svm_pred1=svm_pred(:,i);
TF=[];
for i=1:m
if testY1(i)==0 & svm_pred1(i)==0
    TF(i,1)=1;
    TF(i,2)=0;
end
if testY1(i)==0 & svm_pred1(i)==1
    TF(i,1)=0;
    TF(i,2)=1;  
end
if testY1(i)==1 & svm_pred1(i)==0
    TF(i,1)=0;
    TF(i,2)=1; 
end
if testY1(i)==1 & svm_pred1(i)==1
    TF(i,1)=1;
    TF(i,2)=0;
end
accTP(i,1)=sum(TF(1:i,1)==1);
accFP(i,1)=sum(TF(1:i,2)==1);
Precision(i,1)=accTP(i,1)/(accTP(i,1)+accFP(i,1));
Recall(i,1)=accTP(i,1)/m;
end
PR=[prob_estimates1(:,1) TF accTP accFP Precision Recall];
SPR = sortrows(PR,-1);
AP=mean(Precision)*100;
aP=[aP AP];
end
aP=mean(aP);