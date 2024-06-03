function [FPR, TPR_REC] = cal_pre_rec(pred_now,ground_truth)
%% 开始构造混淆矩阵
label=unique(ground_truth);  % [1,2] 一列两行
type_num = size(label,1);
data=[pred_now ground_truth];           %前面预测标签，后面实际标签
confmat=zeros(type_num);                 %%初始化混淆矩阵
class=cell(1,type_num);

%将数据按照实际标签分开
for i=1:type_num
    in=find(label(i,1)==ground_truth);
    class{1,i}=data(in,:);
end

% 填充混淆矩阵
for i=1:type_num
    for j=1:type_num
        in = find(class{1,i}(:,1)==label(j,1));
        num_in=size(in,1);
        confmat(i,j)=num_in;
    end
end

%% 仅用于疾病二分类的混淆矩阵（0代表正常，1代表有病）0和1 可根据需求来改变。如果不是疾病类形数据可直接去掉 
% 我这边把 0和1 都各自加了1,变成1和2, 因此就将2代表为有病的.

TP=confmat(2,2);
TN=confmat(1,1);
FP=confmat(1,2);
FN=confmat(2,1);

% confmat=[TP FN;FP TN];
if TP==0 & FP==0
     TPR_REC = 0;
     FPR = FP/(FP + TN)    % neg类中有多少被预测为pos    横坐标
else 
     TPR_REC=TP/(TP + FN);     % 正类的召回率  代表纵坐标
     FPR = FP/(FP + TN)    % neg类中有多少被预测为pos    横坐标
end
end

