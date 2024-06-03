
function [train_pvalue,test_pvalue] = p_value(train_data_r,test_data_r,type_num)
%����
    %train_dataΪѵ����
    %verify_dataΪ��֤��
    %test_dataΪ����
%���
    %train_pvalueΪ����ѡ����ѵ����
    %verify_pvalue Ϊ����ѡ������֤��
    %test_pvalueΪ����ѡ���Ĳ��Լ�
    
alpha = 0.05; %������ˮƽ
 
[m1,n1] = size(train_data_r);
[m3,n3] = size(test_data_r);

train_label = train_data_r(:,n1);
data_train = train_data_r(:,2:n1-1); 
data_test = test_data_r(:,2:n3-1);
P =[];
%    ind1 = find(train_label == 1);
%     ind2 = find(train_label == 3);
%     data1 = data_train(ind1,:);
%     data2 = data_train(ind2,:);
%     [h,p] = ttest2(data1,data2,alpha);
%     ind = find(p <= alpha);
for i = 1:type_num-1
    for j = i+1:type_num
        ind1 = find(train_label == i);
        ind2 = find(train_label == j);
        data1 = data_train(ind1,:);
        data2 = data_train(ind2,:);
        [h,p] = ttest2(data1,data2,alpha); %˫����t����
        P = [P;p];
    end
end
mean_p = mean(P);
ind = find(mean_p <= mean(mean_p));
train_pvalue = [train_data_r(:,1) data_train(:,ind)];
test_pvalue = [test_data_r(:,1) data_test(:,ind)];


