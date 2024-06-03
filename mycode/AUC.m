%AUC
function [mean_auc]=AUC(testY,svm_pred,m,type_num)
all_auc=0;
for i=1:type_num
testY1=testY(:,i);svm_pred1=svm_pred(:,i);
[f,~]=size(find(svm_pred1==testY1));
[pred1,~]=size(find(svm_pred1==1));
f0=0;%TN
f1=0;%TP
f2=0;%FP
f3=0;%FN
for j=1:m
    if svm_pred1(j,:)==testY1(j,:)
        if svm_pred1(j,:)==1 %TP
            f1=f1+1;
        else%TN
            f0=f0+1;
        end
     else
        if svm_pred1(j,:)==0
            f3=f3+1;%   FN
        else
            f2=f2+1;%  FP
        end
    end
end
acc=f/m;
sen=f1/(f1+f3);%recall=sen
spe=f0/(f0+f2);
% precision=f1/pred1;
% F_measure=(2*precision*sen)/(precision+sen);
auc=(sen+spe)/2*100;
% G_mean=sqrt(sen*spe)
all_auc=all_auc+auc;
end
mean_auc=all_auc/i;