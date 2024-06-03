function deep_feature = coding(deepnet,m, X)
%coding �Զ������������������ȡ����
%   deepnet ѵ���õĶ�ջ������磻 mΪ��Ҫ�����������Ŀ�� XΪ�������ݣ�������ǩ��
                
    b1 = repmat(deepnet.b{1,1}',m,1);
    encode1 = logsig(X*deepnet.IW{1,1}' + b1); %��һ�����
    

    b2 = repmat(deepnet.b{2,1}',m,1);
    encode2 = logsig(encode1*deepnet.LW{2,1}' + b2); %�ڶ������

    b3 = repmat(deepnet.b{3,1}',m,1);
    encode3 = logsig(encode2*deepnet.LW{3,2}' + b3); %��������� 
    deep_feature = encode3;
end

