function out = featureChoose(features, i);
%UNTITLED4 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    [m, n] = size(features);%����������Ŀ������������Ŀ
    out = zeros(i,n);
    variance = std(features,0,2);%�������׼ƫ��
    [sorted_variance,index] = sort(variance,'descend');%index�����Ԫ����ԭ�����е���λ�û���λ�õ�����
    
%     a = randperm(m);
%     for j=1:i
%        out(j,:) = features(a(j),:); 
%     end
    for j = 1:i
        out(j,:) = features(index(j),:);
    end
end

