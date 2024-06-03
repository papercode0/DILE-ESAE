
%% 
% �ú�������������ַ��� 
% Ϊ��ʹ��ͼ��ʾ��һ��ͼ��,�������ķ���ֵ��Ϊ �������,����������һ��ͼ.���ڸú����л�ͼ��.

%% % ��һ�ַ���
% �����predict �������ΪԤ��Ϊ�������ĸ��ʡ� 

% function  [FPR ,TPR_Rec ] = plot_roc1(predict, ground_truth)  
% % INPUTS  
% %  predict       - �������Բ��Լ��ķ�����  
% %  ground_truth - ���Լ�����ȷ��ǩ,����ֻ���Ƕ����࣬��0��1  
% % OUTPUTS  
% %  auc            - ����ROC���ߵ������µ����  
%   
% %��ʼ��Ϊ��1.0, 1.0��  
% x = 1.0;  
% y = 1.0;  
% 
% %�����ground_truth������������Ŀpos_num�͸���������Ŀneg_num  
% pos_num = sum(ground_truth==2);    % �����1 Ҫ����-1�� 0����1 
% neg_num = sum(ground_truth==1);      
% 
% %���ݸ���Ŀ���Լ������x�����y��Ĳ���  
% x_step = 1.0/neg_num;  
% y_step = 1.0/pos_num;  
% 
% %���ȶ�predict�еķ��������ֵ���մ�С��������  
% [predict,index] = sort(predict);  
% ground_truth = ground_truth(index);    
% 
% %��predict�е�ÿ�������ֱ��ж�������FP������TP  
% %����ground_truth��Ԫ�أ� 
% 
% %��ground_truth[i]=1,��TP������1����y�᷽���½�y_step  
% %��ground_truth[i]=0,��FP������1����x�᷽���½�x_step  
% for i=1:length(ground_truth)  
%     if ground_truth(i) == 2     % �����2 ����positive ?  
%         y = y - y_step;   % ����Ԥ������Ǵ�С�������еģ���Ԥ��Ϊ2�ĸ����Ǵ�С����������2����ô����Ԥ����ˣ�����TP����һ�� 
%                                   %  Ԥ����ˣ�������2��Ԥ���1 ���������ٻ��ʱ�С ��
%     else 
%         x = x - x_step;     % �����С����Ϊ��ʵ��Ϊ1 ��Ԥ��ҲΪ1����ô��û�а�neg Ԥ��Ϊpos������Ҫ��ȥ1������ǰ�neg Ԥ��Ϊpos . ������Ҫ��1 ��
%     end  
%     X(i)=x;  
%     Y(i)=y;  
% end
% 
%  FPR        = [ 1 X ];       %���ϳ�ʼ��  X��
%  TPR_Rec = [ 1 Y] ;       %���ϳ�ʼ��  Y��
%       
% end  


%% �������ַ���ûѧ��

function  [X,Y ] = plot_roc1(predict, ground_truth,svm_predict1)

    %��ʼ��Ϊ��1.0, 1.0��  
%     x = 1.0;  
%     y = 1.0;  
  
   data = [ground_truth svm_predict1]
    %�����ground_truth������������Ŀpos_num�͸���������Ŀneg_num  
    pos_num = sum(ground_truth==1);  
    neg_num = sum(ground_truth==0);  

    %���ݸ���Ŀ���Լ������x�����y��Ĳ���  
%     x_step = 1.0/neg_num;  
%     y_step = 1.0/pos_num;  

    %���ȶ�predict�еķ��������ֵ���մ�С��������  
    [predict,index] = sort(predict);  
    ground_truth = ground_truth(index);  
    
    score=0:0.05:1;

    for i = 1:length(score)
        pred_now = ones(pos_num+neg_num,0);
        in = find(predict>score(1,i));                                                 
        pred_now(in,1) = 1;
        [X(i),Y(i)] = cal_pre_rec(pred_now,ground_truth);
        
    end 
    
%     X=[1 X];%���ϳ�ʼ��
%     Y=[1 Y];%���ϳ�ʼ��
    
%     plot(X,Y,'-ro','LineWidth',2,'MarkerSize',3);hold on;grid on;
%     xlabel('�����ʣ��޲�����Ϊ�в��ı�����');  
%     ylabel('�ٻ��ʣ������в�������Ԥ��Եı�����');  
%     title('ROC����ͼ');  
end
