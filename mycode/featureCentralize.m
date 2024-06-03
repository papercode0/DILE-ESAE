function [X_norm, mu, sigma] = featureCentralize(X)

mu = mean(X);
X_norm = bsxfun(@minus, X, mu);%��������Ԫ���������Ķ�ֵ��������һ�������Ǻ��������ȥ��ֵ

sigma = std(X);%  standard deviation(��׼��)
X_norm = bsxfun(@rdivide, X_norm, sigma);%��������׼����ʹ�������̬�ֲ�

end
