function [new_data,new_label]=KN_datacreat(data,label,k)
%data数据 行样本 列特征
%label数据标签 nx1
%class数据类别个数
%k近邻个数 实际上是算上自己 近邻k-1

%w属性的重要程度
[m,n]=size(data);
D1_index=find(label==1);%找到类别索引
D2_index=find(label==2);
D3_index=find(label==3);%找到类别索引
D4_index=find(label==4);
D5_index=find(label==5);%找到类别索引
D6_index=find(label==6);
D7_index=find(label==7);%找到类别索引
D8_index=find(label==8);
D9_index=find(label==9);%找到类别索引
D10_index=find(label==10);
D11_index=find(label==11);%找到类别索引
D12_index=find(label==12);
D13_index=find(label==13);%找到类别索引
D14_index=find(label==14);
D15_index=find(label==15);%找到类别索引
D16_index=find(label==16);
D17_index=find(label==17);%找到类别索引
D18_index=find(label==18);
D19_index=find(label==19);%找到类别索引
D20_index=find(label==20);
D21_index=find(label==21);
D22_index=find(label==22);
D23_index=find(label==23);
D24_index=find(label==24);
D25_index=find(label==25);
D26_index=find(label==26);

D1=data(D1_index,:);%按类别分出数据集
D2=data(D2_index,:);
D3=data(D3_index,:);
D4=data(D4_index,:);
D5=data(D5_index,:);
D6=data(D6_index,:);
D7=data(D7_index,:);
D8=data(D8_index,:);
D9=data(D9_index,:);
D10=data(D10_index,:);
D11=data(D11_index,:);
D12=data(D12_index,:);
D13=data(D13_index,:);
D14=data(D14_index,:);
D15=data(D15_index,:);
D16=data(D16_index,:);
D17=data(D17_index,:);
D18=data(D18_index,:);
D19=data(D19_index,:);
D20=data(D20_index,:);
D21=data(D21_index,:);
D22=data(D22_index,:);
D23=data(D23_index,:);
D24=data(D24_index,:);
D25=data(D25_index,:);
D26=data(D26_index,:);
new_data=[];
new_label=[];
for i=1:m
dest=data(i,:);%选择一个样本
if label(i,1)==1
    nr=pdist2(dest,D1);%计算该样本和同类样本的欧式距离 1xn hxn  结果 1xh
    [num,idx]=sort(nr);%排序
    spcdata=[D1(idx(2:k),:) D1(idx(k+1),:) D1(idx(k+2),:)];% 
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)]
end
    if label(i,1)==2
    nr=pdist2(dest,D2);
    [num,idx]=sort(nr);%排序
    spcdata=[D2(idx(2:k),:) D2(idx(k+1),:) D2(idx(k+2),:)];%
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*2];
    end
 if label(i,1)==3
    nr=pdist2(dest,D3);
    [num,idx]=sort(nr);%排序
    spcdata=[D3(idx(2:k),:) D3(idx(k+1),:) D3(idx(k+2),:)];% D3(idx(k+1),:)
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*3];
 end
  if label(i,1)==4
    nr=pdist2(dest,D4);
    [num,idx]=sort(nr);%排序
    spcdata=[D4(idx(2:k),:) D4(idx(k+1),:) D4(idx(k+2),:)];% D4(idx(k+1),:)
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*4];
  end
   if label(i,1)==5
    nr=pdist2(dest,D5);
    [num,idx]=sort(nr);%排序
    spcdata=[D5(idx(2:k),:) D5(idx(k+1),:) D5(idx(k+2),:)];% D5(idx(k+1),:)%拼接多少个
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*5];
   end
   if label(i,1)==6
    nr=pdist2(dest,D6);
    [num,idx]=sort(nr);%排序
    spcdata=[D6(idx(2:k),:) D6(idx(k+1),:) D6(idx(k+2),:)];% D6(idx(k+1),:)
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*6];
   end
   if label(i,1)==7
    nr=pdist2(dest,D7);
    [num,idx]=sort(nr);%排序
    spcdata=[D7(idx(2:k),:) D7(idx(k+1),:) D7(idx(k+2),:)];%
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*7];
  end
   if label(i,1)==8
    nr=pdist2(dest,D8);
    [num,idx]=sort(nr);%排序
    spcdata=[D8(idx(2:k),:) D8(idx(k+1),:) D8(idx(k+2),:)];% 
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*8];
   end
   if label(i,1)==9
    nr=pdist2(dest,D9);
    [num,idx]=sort(nr);%排序
    spcdata=[D9(idx(2:k),:) D9(idx(k+1),:) D9(idx(k+2),:)];%
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*9];
   end
    if label(i,1)==10
    nr=pdist2(dest,D10);
    [num,idx]=sort(nr);%排序
    spcdata=[D10(idx(2:k),:) D10(idx(k+1),:) D10(idx(k+2),:)];% 
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*10];
    end
     if label(i,1)==11
    nr=pdist2(dest,D11);
    [num,idx]=sort(nr);%排序
    spcdata=[D11(idx(2:k),:) D11(idx(k+1),:) D11(idx(k+2),:)];%
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*11];
     end
     if label(i,1)==12
    nr=pdist2(dest,D12);
    [num,idx]=sort(nr);%排序
    spcdata=[D12(idx(2:k),:) D12(idx(k+1),:) D12(idx(k+2),:)];%
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*12];
     end
     if label(i,1)==13
    nr=pdist2(dest,D13);
    [num,idx]=sort(nr);%排序
    spcdata=[D13(idx(2:k),:) D13(idx(k+1),:) D13(idx(k+2),:)];%
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*13];
     end
     if label(i,1)==14
    nr=pdist2(dest,D14);
    [num,idx]=sort(nr);%排序
    spcdata=[D14(idx(2:k),:) D14(idx(k+1),:) D14(idx(k+2),:)];%
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*14];
     end
     if label(i,1)==15
    nr=pdist2(dest,D15);
    [num,idx]=sort(nr);%排序
    spcdata=[D15(idx(2:k),:) D15(idx(k+1),:) D15(idx(k+2),:)];%
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*15];
     end
     if label(i,1)==16
    nr=pdist2(dest,D16);
    [num,idx]=sort(nr);%排序
    spcdata=[D16(idx(2:k),:) D16(idx(k+1),:) D16(idx(k+2),:)];%
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*16];
     end
     if label(i,1)==17
    nr=pdist2(dest,D17);
    [num,idx]=sort(nr);%排序
    spcdata=[D17(idx(2:k),:) D17(idx(k+1),:) D17(idx(k+2),:)];%
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*17];
     end
     if label(i,1)==18
    nr=pdist2(dest,D18);
    [num,idx]=sort(nr);%排序
    spcdata=[D18(idx(2:k),:) D18(idx(k+1),:) D18(idx(k+2),:)];%
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*18];
     end
     if label(i,1)==19
    nr=pdist2(dest,D19);
    [num,idx]=sort(nr);%排序
    spcdata=[D19(idx(2:k),:) D19(idx(k+1),:) D19(idx(k+2),:)];%
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*19];
     end
     if label(i,1)==20
    nr=pdist2(dest,D20);
    [num,idx]=sort(nr);%排序
    spcdata=[D20(idx(2:k),:) D20(idx(k+1),:) D20(idx(k+2),:)];%
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*20];
     end
     if label(i,1)==21
    nr=pdist2(dest,D21);
    [num,idx]=sort(nr);%排序
    spcdata=[D21(idx(2:k),:) D21(idx(k+1),:) D21(idx(k+2),:)];%
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*21];
     end
     if label(i,1)==22
    nr=pdist2(dest,D22);
    [num,idx]=sort(nr);%排序
    spcdata=[D22(idx(2:k),:) D22(idx(k+1),:) D22(idx(k+2),:)];%
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*22];
     end
     if label(i,1)==23
    nr=pdist2(dest,D23);
    [num,idx]=sort(nr);%排序
    spcdata=[D23(idx(2:k),:) D23(idx(k+1),:) D23(idx(k+2),:)];%
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*23];
     end
     if label(i,1)==24
    nr=pdist2(dest,D24);
    [num,idx]=sort(nr);%排序
    spcdata=[D24(idx(2:k),:) D24(idx(k+1),:) D24(idx(k+2),:)];%
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*24];
     end
     
     if label(i,1)==25
    nr=pdist2(dest,D25);
    [num,idx]=sort(nr);%排序
    spcdata=[D25(idx(2:k),:) D25(idx(k+1),:) D25(idx(k+2),:)];%
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*25];
     end
     
  if label(i,1)==26
    nr=pdist2(dest,D26);
    [num,idx]=sort(nr);%排序
    spcdata=[D26(idx(2:k),:) D26(idx(k+1),:) D26(idx(k+2),:)];%
    new_data=[new_data;spcdata];
    new_label=[new_label;ones(k-1,1)*26];
     end
 
end
end