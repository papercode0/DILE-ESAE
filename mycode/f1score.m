    
function  [F1] = f1score(type_num,svm_pred,testY)
[A,~]=confusionmat(svm_pred,testY);
    c=type_num;
    sum1=0;
    for l=1:c
        precise(l)=A(l,l)/sum(A(:,l));
        recall(l)=A(l,l)/sum(A(l,:));
    end
   Precise=sum(precise)/c;
   Recall=sum(recall)/c;
   F1 = 2 * Precise* Recall/(Precise + Recall)*100;  