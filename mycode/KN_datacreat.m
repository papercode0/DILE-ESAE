function [new_data]=KN_datacreat(train_data,train_label,test_data,test_label,k)
%data���� ������ ������
%label���ݱ�ǩ nx1
%class����������
%k���ڸ��� ʵ�����������Լ� ����k-1

%w���Ե���Ҫ�̶�
[m,n]=size(train_data);
[g,h]=size(test_data);
D1_index=find(train_label==1);%�ҵ��������
D2_index=find(train_label==2);
D3_index=find(train_label==3);%�ҵ��������
D4_index=find(train_label==4);
D5_index=find(train_label==5);%�ҵ��������
D6_index=find(train_label==6);
D7_index=find(train_label==7);%�ҵ��������
D8_index=find(train_label==8);
D9_index=find(train_label==9);%�ҵ��������
D10_index=find(train_label==10);
D11_index=find(train_label==11);%�ҵ��������
D12_index=find(train_label==12);
D13_index=find(train_label==13);%�ҵ��������
D14_index=find(train_label==14);
D15_index=find(train_label==15);%�ҵ��������
D16_index=find(train_label==16);
D17_index=find(train_label==17);%�ҵ��������
D18_index=find(train_label==18);
D19_index=find(train_label==19);%�ҵ��������
D20_index=find(train_label==20);
D21_index=find(train_label==21);
D22_index=find(train_label==22);
D23_index=find(train_label==23);
D24_index=find(train_label==24);
D25_index=find(train_label==25);
D26_index=find(train_label==26);

D1=train_data(D1_index,:);%�����ֳ����ݼ�
D2=train_data(D2_index,:);
D3=train_data(D3_index,:);
D4=train_data(D4_index,:);
D5=train_data(D5_index,:);
D6=train_data(D6_index,:);
D7=train_data(D7_index,:);
D8=train_data(D8_index,:);
D9=train_data(D9_index,:);
D10=train_data(D10_index,:);
D11=train_data(D11_index,:);
D12=train_data(D12_index,:);
D13=train_data(D13_index,:);
D14=train_data(D14_index,:);
D15=train_data(D15_index,:);
D16=train_data(D16_index,:);
D17=train_data(D17_index,:);
D18=train_data(D18_index,:);
D19=train_data(D19_index,:);
D20=train_data(D20_index,:);
D21=train_data(D21_index,:);
D22=train_data(D22_index,:);
D23=train_data(D23_index,:);
D24=train_data(D24_index,:);
D25=train_data(D25_index,:);
D26=train_data(D26_index,:);

new_data=[];
new_label=[];
for j=1:g %��������
   
dest=test_data(j,:);%ѡ��һ������
if test_label(j,1)==1   %�жϲ����������
%      for i=1:m 
% if train_label(i,1)==1
    nr=pdist2(dest,D1);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
% end
%      end

    new_data=[new_data;D1(idx(1:k),:)];
%     new_label=[new_label;ones(k-1,1)]
end
if test_label(j,1)==2
%      for i=1:m 
% if train_label(i,1)==2
    nr=pdist2(dest,D2);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
     [num,idx]=sort(nr);%����
% end
%      end

    new_data=[new_data;D2(idx(1:k),:)];
%     new_label=[new_label;ones(k-1,1)*2]
end
if test_label(j,1)==3
%      for i=1:m 
% if train_label(i,1)==3
    nr=pdist2(dest,D3);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
     [num,idx]=sort(nr);%����
% end
%      end
    new_data=[new_data;D3(idx(1:k),:)];
%     new_label=[new_label;ones(k-1,1)*3]
end

if test_label(j,1)==4   %�жϲ����������
    nr=pdist2(dest,D4);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D4(idx(1:k),:)];
end

if test_label(j,1)==5   %�жϲ����������
    nr=pdist2(dest,D5);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D5(idx(1:k),:)];
end

if test_label(j,1)==6   %�жϲ����������
    nr=pdist2(dest,D6);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D6(idx(1:k),:)];
end

if test_label(j,1)==7   %�жϲ����������
    nr=pdist2(dest,D7);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D7(idx(1:k),:)];
end
if test_label(j,1)==8   %�жϲ����������
   nr=pdist2(dest,D8);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D8(idx(1:k),:)];
end
if test_label(j,1)==9   %�жϲ����������
    nr=pdist2(dest,D9);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D9(idx(1:k),:)];
end

if test_label(j,1)==10   %�жϲ����������
    nr=pdist2(dest,D10);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D10(idx(1:k),:)];
end

if test_label(j,1)==11   %�жϲ����������
    nr=pdist2(dest,D11);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D11(idx(1:k),:)];
end

if test_label(j,1)==12   %�жϲ����������
    nr=pdist2(dest,D12);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D12(idx(1:k),:)];
end

if test_label(j,1)==13   %�жϲ����������
    nr=pdist2(dest,D13);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D13(idx(1:k),:)];
end

if test_label(j,1)==14   %�жϲ����������
    nr=pdist2(dest,D14);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D14(idx(1:k),:)];
end

if test_label(j,1)==15   %�жϲ����������
    nr=pdist2(dest,D15);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D15(idx(1:k),:)];
end

if test_label(j,1)==16   %�жϲ����������
    nr=pdist2(dest,D16);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D16(idx(1:k),:)];
end

if test_label(j,1)==17   %�жϲ����������
    nr=pdist2(dest,D17);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D17(idx(1:k),:)];
end

if test_label(j,1)==18   %�жϲ����������
    nr=pdist2(dest,D18);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D18(idx(1:k),:)];
end

if test_label(j,1)==19   %�жϲ����������
    nr=pdist2(dest,D19);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D19(idx(1:k),:)];
end

if test_label(j,1)==20   %�жϲ����������
    nr=pdist2(dest,D20);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D20(idx(1:k),:)];
end

if test_label(j,1)==21   %�жϲ����������
    nr=pdist2(dest,D21);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D21(idx(1:k),:)];
end

if test_label(j,1)==22   %�жϲ����������
    nr=pdist2(dest,D22);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D22(idx(1:k),:)];
end

if test_label(j,1)==23   %�жϲ����������
    nr=pdist2(dest,D23);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D23(idx(1:k),:)];
end

if test_label(j,1)==24   %�жϲ����������
    nr=pdist2(dest,D24);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D24(idx(1:k),:)];
end

if test_label(j,1)==25   %�жϲ����������
    nr=pdist2(dest,D25);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D25(idx(1:k),:)];
end

if test_label(j,1)==26   %�жϲ����������
    nr=pdist2(dest,D26);%�����������ѵ������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
       [num,idx]=sort(nr);%����
    new_data=[new_data;D26(idx(1:k),:)];
end
%     if train_label(i,1)==2
%     nr=pdist2(dest,Dni);
%     [num,idx]=sort(nr);%����
%     new_data=[new_data;Dni(idx(2:k),:)];
%     new_label=[new_label;ones(k-1,1)*2];
%     end
%  if train_label(i,1)==3
%     nr=pdist2(dest,D3);
%     [num,idx]=sort(nr);%����
%     new_data=[new_data;D3(idx(2:k),:)];
%     new_label=[new_label;ones(k-1,1)*3];
%  end
%   if train_label(i,1)==4
%     nr=pdist2(dest,D4);
%     [num,idx]=sort(nr);%����
%     new_data=[new_data;D4(idx(2:k),:)];
%     new_label=[new_label;ones(k-1,1)*4];
%   end
%    if train_label(i,1)==5
%     nr=pdist2(dest,D4);
%     [num,idx]=sort(nr);%����
%     new_data=[new_data;D4(idx(2:k),:)];
%     new_label=[new_label;ones(k-1,1)*5];
%    end
%    if train_label(i,1)==6
%     nr=pdist2(dest,D4);
%     [num,idx]=sort(nr);%����
%     new_data=[new_data;D4(idx(2:k),:)];
%     new_label=[new_label;ones(k-1,1)*6];
%    end
%    if train_label(i,1)==7
%     nr=pdist2(dest,D4);
%     [num,idx]=sort(nr);%����
%     new_data=[new_data;D4(idx(2:k),:)];
%     new_label=[new_label;ones(k-1,1)*7];
%   end
%    if train_label(i,1)==8
%     nr=pdist2(dest,D4);
%     [num,idx]=sort(nr);%����
%     new_data=[new_data;D4(idx(2:k),:)];
%     new_label=[new_label;ones(k-1,1)*8];
%    end
%    if train_label(i,1)==9
%     nr=pdist2(dest,D4);
%     [num,idx]=sort(nr);%����
%     new_data=[new_data;D4(idx(2:k),:)];
%     new_label=[new_label;ones(k-1,1)*9];
%    end
%     if train_label(i,1)==10
%     nr=pdist2(dest,D4);
%     [num,idx]=sort(nr);%����
%     new_data=[new_data;D4(idx(2:k),:)];
%     new_label=[new_label;ones(k-1,1)*10];
%  end
% end
%     end
end
end