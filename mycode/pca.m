function [U, S] = pca(X)

[m, n] = size(X);  %m��n��,�б�ʾ������

U = zeros(n);   %����nά0����
S = zeros(n);

Sigma = X'*X;
[U,S,V] = svd(Sigma);%����ֵ�ֽ�  Sigma=U*S*V'

end
