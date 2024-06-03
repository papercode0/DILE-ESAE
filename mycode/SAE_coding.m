function deep_feature = SAE_coding(deepnet,m, X)
%coding �Զ������������������ȡ����
%   deepnet ѵ���õĶ�ջ������磻 mΪ��Ҫ�����������Ŀ�� XΪ�������ݣ�������ǩ��
                
    b1 = repmat(deepnet.b{1,1}',m,1);
    encode = logsig(X*deepnet.IW{1,1}' + b1); %��һ�����
    deep_feature = encode;
end
