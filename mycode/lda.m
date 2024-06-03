function [U, S]=lda(X,Y,K)   % K为样本类别数目
[m, n] = size(X);   
U = zeros(n);  
S = zeros(n);  
centroids = zeros(K,n);
lengthCindex = zeros(1,K);%存放K个类样本数目
%% 寻找类内中心
for i = 1:K
cindex = find(Y == i);  %返回一个向量，该向量由第i类的数据所在行脚标构成
lengthCindex(i) = size(cindex,1); %cindex的行数即第i类样本数目
    for j = 1:lengthCindex(i)
    centroids(i,:) = X(cindex(j),:)+centroids(i,:); 
    end
centroids(i,:) = centroids(i,:)/lengthCindex(i); %类内中心
end

SB=zeros(n,n); %计算类间散度矩阵
mu = mean(X);
for i=1:K
SB=SB+lengthCindex(i)*(centroids(i,:)-mu)'*(centroids(i,:)-mu);
end

SW=zeros(n,n); 
for i=1:K
    cindex = find(Y == i);
    lengthCindex(i) = size(cindex,1);
    for j = 1:lengthCindex(i) 
        SW=SW+(X(cindex(j),:)-centroids(i,:))'*(X(cindex(j),:)-centroids(i,:));
    end
end

matrix=pinv(SW)*SB;  %伪逆
[U,S,V] = svd(matrix);  %奇异值分解

end

