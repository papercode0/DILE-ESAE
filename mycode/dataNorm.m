function [trNorm, ttNorm] = dataNorm(trData, ttData)
%���ڶ�ѵ�����Ͳ��Լ���һ��
%���һ���Ǳ�ǩ
t1 = trData(:, 1:(end-1));
t2 = trData(:, end);
t3 = ttData(:, 1:(end-1));
t4 = ttData(:, end);

[t5, ps] = mapminmax(t1',0.1, 1);
trNorm = [t5', t2];
t6 = mapminmax('apply', t3', ps);
ttNorm = [t6', t4];