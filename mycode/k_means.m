function [cluster_X,cluster_Y] = k_means(trainX,trainY,type_num,p)
%sampleCluster �������ຯ�������ھ��ࣩ  2/3, 1/2, 1/3, 1/4
cluster_X = [];
cluster_Y = [];
for i = 1:type_num
    % ��ѡ��i������
    idx = find(trainY==i);
    [m,n] = size(idx);
    temp = [];
    for j = 1:m 
        temp = [temp;trainX(idx(j),:)]; 
    end
    % ����i����������Ϊԭ����һ��
    [idx,temp] = kmeans(temp,floor(m*p));
    label = ones(floor(m*p),1);
    label(:) = i;
    %��ӱ�ǩ
    cluster_Y = [cluster_Y;label];
    %�õ��µ�ѵ����
    cluster_X = [cluster_X;temp];
end

end


