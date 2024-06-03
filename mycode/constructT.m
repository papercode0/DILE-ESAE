function [T] = constructT(Y,type_num)
%constructT ����one-hot��ǩ��ʽ
% type_num = size(unique(Y),1);%�����Ŀ��Y�м���������ͬ��Ԫ�أ�
[m,n] = size(Y);
T = zeros(type_num,m);
for i = 1:type_num
   index = find(Y==i);
   c = size(index,1);
   for j = 1:c
       T(i,index(j)) = 1;
   end
end
 
end

