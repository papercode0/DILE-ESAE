function [U,S] = lpp(X,W) 

[m, n] = size(X);
U = zeros(n,n);
S = zeros(n,n);

D = diag(sum(W,2));  %亲疏矩阵每行元素求和构成对角矩阵D
L = D - W;        %Laplacian矩阵
T1 = X'*L*X;      
T2 = X'*D*X;
T = pinv(T2)*T1;%pinv求T2的伪逆

[U,S,V] = svd(T);%奇异值分解
[val,ind] = sort(diag(S));%%求解特征值，并将其排序，val表示排序后的值，ind表示排序后的数据在原始数据中的位置
U = U(:,ind);  %按照排序后特征值在原始数据中的位置来对特征向量重新排序
S = S(ind,:);  

end
