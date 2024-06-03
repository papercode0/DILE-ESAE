function deep_feature = SAE_coding(deepnet,m, X)
%coding 自动编码器的深度特征提取函数
%   deepnet 训练好的堆栈深度网络； m为需要编码的样本数目， X为编码数据（不含标签）
                
    b1 = repmat(deepnet.b{1,1}',m,1);
    encode = logsig(X*deepnet.IW{1,1}' + b1); %第一层输出
    deep_feature = encode;
end
