function [U,S] = lpp(X,W) 

[m, n] = size(X);
U = zeros(n,n);
S = zeros(n,n);

D = diag(sum(W,2));  %�������ÿ��Ԫ����͹��ɶԽǾ���D
L = D - W;        %Laplacian����
T1 = X'*L*X;      
T2 = X'*D*X;
T = pinv(T2)*T1;%pinv��T2��α��

[U,S,V] = svd(T);%����ֵ�ֽ�
[val,ind] = sort(diag(S));%%�������ֵ������������val��ʾ������ֵ��ind��ʾ������������ԭʼ�����е�λ��
U = U(:,ind);  %�������������ֵ��ԭʼ�����е�λ����������������������
S = S(ind,:);  

end
