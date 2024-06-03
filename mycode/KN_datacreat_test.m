function [new_data]=KN_datacreat_test(train_data,train_label,test_data,test_label,k)
%data数据 行样本 列特征
%label数据标签 nx1
%class数据类别个数
%k近邻个数 实际上是算上自己 近邻k-1

%w属性的重要程度
[m,n]=size(train_data);
[g,h]=size(test_data);
D1_index=find(train_label==1);%找到类别索引
D2_index=find(train_label==2);
D3_index=find(train_label==3);%找到类别索引
D4_index=find(train_label==4);
D5_index=find(train_label==5);%找到类别索引
D6_index=find(train_label==6);
D7_index=find(train_label==7);%找到类别索引
D8_index=find(train_label==8);
D9_index=find(train_label==9);%找到类别索引
D10_index=find(train_label==10);
D11_index=find(train_label==11);%找到类别索引
D12_index=find(train_label==12);
D13_index=find(train_label==13);%找到类别索引
D14_index=find(train_label==14);
D15_index=find(train_label==15);%找到类别索引
D16_index=find(train_label==16);
D17_index=find(train_label==17);%找到类别索引
D18_index=find(train_label==18);
D19_index=find(train_label==19);%找到类别索引
D20_index=find(train_label==20);
D21_index=find(train_label==21);
D22_index=find(train_label==22);
D23_index=find(train_label==23);
D24_index=find(train_label==24);
D25_index=find(train_label==25);
D26_index=find(train_label==26);

D1=train_data(D1_index,:);%按类别分出数据集
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
for j=1:g %测试样本
   
dest=test_data(j,:);%选择一个样本
if test_label(j,1)==1   %判断测试样本类别
%      for i=1:m 
% if train_label(i,1)==1
    nr=pdist2(dest,D1);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
% end
%      end
    spcdata=[D1(idx(1:k),:) D1(idx(k+1),:) D1(idx(k+2),:)];
    new_data=[new_data;spcdata];
%     new_label=[new_label;ones(k-1,1)]
end
if test_label(j,1)==2
%      for i=1:m 
% if train_label(i,1)==2
    nr=pdist2(dest,D2);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
     [num,idx]=sort(nr);%排序
% end
%      end

    spcdata=[D2(idx(1:k),:) D2(idx(k+1),:) D2(idx(k+2),:)];
    new_data=[new_data;spcdata];
%     new_label=[new_label;ones(k-1,1)*2]
end
if test_label(j,1)==3
%      for i=1:m 
% if train_label(i,1)==3
    nr=pdist2(dest,D3);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
     [num,idx]=sort(nr);%排序
% end
%      end
    spcdata=[D3(idx(1:k),:) D3(idx(k+1),:) D3(idx(k+2),:)];
    new_data=[new_data;spcdata];
%     new_label=[new_label;ones(k-1,1)*3]
end

if test_label(j,1)==4   %判断测试样本类别
    nr=pdist2(dest,D4);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D4(idx(1:k),:) D4(idx(k+1),:) D4(idx(k+2),:)];
    new_data=[new_data;spcdata];
end

if test_label(j,1)==5   %判断测试样本类别
    nr=pdist2(dest,D5);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D5(idx(1:k),:) D5(idx(k+1),:) D5(idx(k+2),:)];
    new_data=[new_data;spcdata];
end

if test_label(j,1)==6   %判断测试样本类别
    nr=pdist2(dest,D6);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D6(idx(1:k),:) D6(idx(k+1),:) D6(idx(k+2),:)];
    new_data=[new_data;spcdata];
end

if test_label(j,1)==7   %判断测试样本类别
    nr=pdist2(dest,D7);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D7(idx(1:k),:) D7(idx(k+1),:) D7(idx(k+2),:)];
    new_data=[new_data;spcdata];
end
if test_label(j,1)==8   %判断测试样本类别
   nr=pdist2(dest,D8);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D8(idx(1:k),:) D8(idx(k+1),:) D8(idx(k+2),:)];
    new_data=[new_data;spcdata];
end
if test_label(j,1)==9   %判断测试样本类别
    nr=pdist2(dest,D9);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D9(idx(1:k),:) D9(idx(k+1),:) D9(idx(k+2),:)];
    new_data=[new_data;spcdata];
end

if test_label(j,1)==10   %判断测试样本类别
    nr=pdist2(dest,D10);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D10(idx(1:k),:) D10(idx(k+1),:) D10(idx(k+2),:)];
    new_data=[new_data;spcdata];
end

if test_label(j,1)==11   %判断测试样本类别
    nr=pdist2(dest,D11);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D11(idx(1:k),:) D11(idx(k+1),:) D11(idx(k+2),:)];
    new_data=[new_data;spcdata];
end

if test_label(j,1)==12   %判断测试样本类别
    nr=pdist2(dest,D12);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D12(idx(1:k),:) D12(idx(k+1),:) D12(idx(k+2),:)];
    new_data=[new_data;spcdata];
end

if test_label(j,1)==13   %判断测试样本类别
    nr=pdist2(dest,D13);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D13(idx(1:k),:) D13(idx(k+1),:) D13(idx(k+2),:)];
    new_data=[new_data;spcdata];
end

if test_label(j,1)==14   %判断测试样本类别
    nr=pdist2(dest,D14);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D14(idx(1:k),:) D14(idx(k+1),:) D14(idx(k+2),:)];
    new_data=[new_data;spcdata];
end

if test_label(j,1)==15   %判断测试样本类别
    nr=pdist2(dest,D15);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D15(idx(1:k),:) D15(idx(k+1),:) D15(idx(k+2),:)];
    new_data=[new_data;spcdata];
end

if test_label(j,1)==16   %判断测试样本类别
    nr=pdist2(dest,D16);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D16(idx(1:k),:) D16(idx(k+1),:) D16(idx(k+2),:)];
    new_data=[new_data;spcdata];
end

if test_label(j,1)==17   %判断测试样本类别
    nr=pdist2(dest,D17);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D17(idx(1:k),:) D17(idx(k+1),:) D17(idx(k+2),:)];
    new_data=[new_data;spcdata];
end

if test_label(j,1)==18   %判断测试样本类别
    nr=pdist2(dest,D18);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D18(idx(1:k),:) D18(idx(k+1),:) D18(idx(k+2),:)];
    new_data=[new_data;spcdata];
end

if test_label(j,1)==19   %判断测试样本类别
    nr=pdist2(dest,D19);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D19(idx(1:k),:) D19(idx(k+1),:) D19(idx(k+2),:)];
    new_data=[new_data;spcdata];
end

if test_label(j,1)==20   %判断测试样本类别
    nr=pdist2(dest,D20);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D20(idx(1:k),:) D20(idx(k+1),:) D20(idx(k+2),:)];
    new_data=[new_data;spcdata];
end

if test_label(j,1)==21   %判断测试样本类别
    nr=pdist2(dest,D21);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D21(idx(1:k),:) D21(idx(k+1),:) D21(idx(k+2),:)];
    new_data=[new_data;spcdata];
end

if test_label(j,1)==22   %判断测试样本类别
    nr=pdist2(dest,D22);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D22(idx(1:k),:) D22(idx(k+1),:) D22(idx(k+2),:)];
    new_data=[new_data;spcdata];
end

if test_label(j,1)==23   %判断测试样本类别
    nr=pdist2(dest,D23);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D23(idx(1:k),:) D23(idx(k+1),:) D23(idx(k+2),:)];
    new_data=[new_data;spcdata];
end

if test_label(j,1)==24   %判断测试样本类别
    nr=pdist2(dest,D24);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D24(idx(1:k),:) D24(idx(k+1),:) D24(idx(k+2),:)];
    new_data=[new_data;spcdata];
end

if test_label(j,1)==25   %判断测试样本类别
    nr=pdist2(dest,D25);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D25(idx(1:k),:) D25(idx(k+1),:) D25(idx(k+2),:)];
    new_data=[new_data;spcdata];
end

if test_label(j,1)==26   %判断测试样本类别
    nr=pdist2(dest,D26);%计算该样本和训练集中同类样本的欧式距离 1xn hxn  结果 1xh
       [num,idx]=sort(nr);%排序
    spcdata=[D26(idx(1:k),:) D26(idx(k+1),:) D26(idx(k+2),:)];
    new_data=[new_data;spcdata];
end
%     if train_label(i,1)==2
%     nr=pdist2(dest,Dni);
%     [num,idx]=sort(nr);%排序
%     new_data=[new_data;Dni(idx(2:k),:)];
%     new_label=[new_label;ones(k-1,1)*2];
%     end
%  if train_label(i,1)==3
%     nr=pdist2(dest,D3);
%     [num,idx]=sort(nr);%排序
%     new_data=[new_data;D3(idx(2:k),:)];
%     new_label=[new_label;ones(k-1,1)*3];
%  end
%   if train_label(i,1)==4
%     nr=pdist2(dest,D4);
%     [num,idx]=sort(nr);%排序
%     new_data=[new_data;D4(idx(2:k),:)];
%     new_label=[new_label;ones(k-1,1)*4];
%   end
%    if train_label(i,1)==5
%     nr=pdist2(dest,D4);
%     [num,idx]=sort(nr);%排序
%     new_data=[new_data;D4(idx(2:k),:)];
%     new_label=[new_label;ones(k-1,1)*5];
%    end
%    if train_label(i,1)==6
%     nr=pdist2(dest,D4);
%     [num,idx]=sort(nr);%排序
%     new_data=[new_data;D4(idx(2:k),:)];
%     new_label=[new_label;ones(k-1,1)*6];
%    end
%    if train_label(i,1)==7
%     nr=pdist2(dest,D4);
%     [num,idx]=sort(nr);%排序
%     new_data=[new_data;D4(idx(2:k),:)];
%     new_label=[new_label;ones(k-1,1)*7];
%   end
%    if train_label(i,1)==8
%     nr=pdist2(dest,D4);
%     [num,idx]=sort(nr);%排序
%     new_data=[new_data;D4(idx(2:k),:)];
%     new_label=[new_label;ones(k-1,1)*8];
%    end
%    if train_label(i,1)==9
%     nr=pdist2(dest,D4);
%     [num,idx]=sort(nr);%排序
%     new_data=[new_data;D4(idx(2:k),:)];
%     new_label=[new_label;ones(k-1,1)*9];
%    end
%     if train_label(i,1)==10
%     nr=pdist2(dest,D4);
%     [num,idx]=sort(nr);%排序
%     new_data=[new_data;D4(idx(2:k),:)];
%     new_label=[new_label;ones(k-1,1)*10];
%  end
% end
%     end
end
end