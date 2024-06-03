function [U, S] = pca(X)

[m, n] = size(X);  %m行n列,列表示特征数

U = zeros(n);   %生成n维0矩阵
S = zeros(n);

Sigma = X'*X;
[U,S,V] = svd(Sigma);%奇异值分解  Sigma=U*S*V'

end
