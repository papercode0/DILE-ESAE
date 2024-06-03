function [FPR, TPR_REC] = cal_pre_rec(pred_now,ground_truth)
%% ��ʼ�����������
label=unique(ground_truth);  % [1,2] һ������
type_num = size(label,1);
data=[pred_now ground_truth];           %ǰ��Ԥ���ǩ������ʵ�ʱ�ǩ
confmat=zeros(type_num);                 %%��ʼ����������
class=cell(1,type_num);

%�����ݰ���ʵ�ʱ�ǩ�ֿ�
for i=1:type_num
    in=find(label(i,1)==ground_truth);
    class{1,i}=data(in,:);
end

% ����������
for i=1:type_num
    for j=1:type_num
        in = find(class{1,i}(:,1)==label(j,1));
        num_in=size(in,1);
        confmat(i,j)=num_in;
    end
end

%% �����ڼ���������Ļ�������0����������1�����в���0��1 �ɸ����������ı䡣������Ǽ����������ݿ�ֱ��ȥ�� 
% ����߰� 0��1 �����Լ���1,���1��2, ��˾ͽ�2����Ϊ�в���.

TP=confmat(2,2);
TN=confmat(1,1);
FP=confmat(1,2);
FN=confmat(2,1);

% confmat=[TP FN;FP TN];
if TP==0 & FP==0
     TPR_REC = 0;
     FPR = FP/(FP + TN)    % neg�����ж��ٱ�Ԥ��Ϊpos    ������
else 
     TPR_REC=TP/(TP + FN);     % ������ٻ���  ����������
     FPR = FP/(FP + TN)    % neg�����ж��ٱ�Ԥ��Ϊpos    ������
end
end

