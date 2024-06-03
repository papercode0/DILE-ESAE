
%% 
% 该函数里面包含两种方法 
% 为了使得图显示在一张图上,将函数的返回值变为 返回左边,在主程序中一起画图.不在该函数中画图了.

%% % 第一种方法
% 这里的predict 可以理解为预测为正样本的概率。 

% function  [FPR ,TPR_Rec ] = plot_roc1(predict, ground_truth)  
% % INPUTS  
% %  predict       - 分类器对测试集的分类结果  
% %  ground_truth - 测试集的正确标签,这里只考虑二分类，即0和1  
% % OUTPUTS  
% %  auc            - 返回ROC曲线的曲线下的面积  
%   
% %初始点为（1.0, 1.0）  
% x = 1.0;  
% y = 1.0;  
% 
% %计算出ground_truth中正样本的数目pos_num和负样本的数目neg_num  
% pos_num = sum(ground_truth==2);    % 这里的1 要换成-1， 0换成1 
% neg_num = sum(ground_truth==1);      
% 
% %根据该数目可以计算出沿x轴或者y轴的步长  
% x_step = 1.0/neg_num;  
% y_step = 1.0/pos_num;  
% 
% %首先对predict中的分类器输出值按照从小到大排列  
% [predict,index] = sort(predict);  
% ground_truth = ground_truth(index);    
% 
% %对predict中的每个样本分别判断他们是FP或者是TP  
% %遍历ground_truth的元素， 
% 
% %若ground_truth[i]=1,则TP减少了1，往y轴方向下降y_step  
% %若ground_truth[i]=0,则FP减少了1，往x轴方向下降x_step  
% for i=1:length(ground_truth)  
%     if ground_truth(i) == 2     % 这里的2 代表positive ?  
%         y = y - y_step;   % 由于预测概率是从小往大排列的（即预测为2的概率是从小到达）如果等于2，那么就是预测错了，代表TP少了一次 
%                                   %  预测错了（本来是2，预测程1 ，即正类召回率变小 ）
%     else 
%         x = x - x_step;     % 这里减小是因为，实际为1 ，预测也为1，那么就没有把neg 预测为pos，所以要减去1，如果是把neg 预测为pos . 横坐标要加1 。
%     end  
%     X(i)=x;  
%     Y(i)=y;  
% end
% 
%  FPR        = [ 1 X ];       %加上初始点  X轴
%  TPR_Rec = [ 1 Y] ;       %加上初始点  Y轴
%       
% end  


%% 下面这种方法没学会

function  [X,Y ] = plot_roc1(predict, ground_truth,svm_predict1)

    %初始点为（1.0, 1.0）  
%     x = 1.0;  
%     y = 1.0;  
  
   data = [ground_truth svm_predict1]
    %计算出ground_truth中正样本的数目pos_num和负样本的数目neg_num  
    pos_num = sum(ground_truth==1);  
    neg_num = sum(ground_truth==0);  

    %根据该数目可以计算出沿x轴或者y轴的步长  
%     x_step = 1.0/neg_num;  
%     y_step = 1.0/pos_num;  

    %首先对predict中的分类器输出值按照从小到大排列  
    [predict,index] = sort(predict);  
    ground_truth = ground_truth(index);  
    
    score=0:0.05:1;

    for i = 1:length(score)
        pred_now = ones(pos_num+neg_num,0);
        in = find(predict>score(1,i));                                                 
        pred_now(in,1) = 1;
        [X(i),Y(i)] = cal_pre_rec(pred_now,ground_truth);
        
    end 
    
%     X=[1 X];%加上初始点
%     Y=[1 Y];%加上初始点
    
%     plot(X,Y,'-ro','LineWidth',2,'MarkerSize',3);hold on;grid on;
%     xlabel('假阳率（无病被认为有病的比例）');  
%     ylabel('召回率（真正有病的人你预测对的比例）');  
%     title('ROC曲线图');  
end
