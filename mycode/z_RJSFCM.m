function [P V steps obj] = RJSFCM(X,W,dataY1, ProK,Pmu,Penta,maxStep,conCriterion)
%% robust jointly sparse fuzzy C-means
% X:数据集， n*m，行向量形式
% W:由LLE导出的相似度矩阵
% ProK：选择的特征数目
%Pgamma，Pmu, Penta：参数
% maxStep：最大迭代步数
% conCriterion ：收敛条件
%P: m*ProK;
%V cluster_n*m  行向量
%U n*cluster_n  行向量
%InitU  cluster_n*n 
% T = constructT(dataY1); 
%%--------------------begin------------------------------------------------
data_n = size(X,1);
dim_n = size(X,2);
options = [2, 50, 1e-5, 0];
%  InitU=dataY1;
expo = -1;
V=dataY1;
%  V=X;
% U = D';    %n*cluster_n 

P = eye(dim_n,ProK);    %初始化为单位阵

steps=0;
converged=false;

while ~converged && steps<=maxStep
     
    steps=steps+1; 
    
    V_Old = V;
    
     %%%%%%%%%%%%%%%%%%%%%（求Uik）  %z:均值聚类没有用到，所以注释了这个过程
%      t1 = clock;
       %%%% （求hik，||XiP-VkP||_2  
      h = EuDist2((X*P),(V*P));   %%距离，已经开方，L21
%  
%      for i=1:1:data_n
%         vi = -1*h(i,:)/(2*Pgamma);
%         U(i,:) = EProjSimplex(vi);
%      end     
%      disp(['U:',num2str(etime(clock,t1))]);
     
    %%%%%%%%%%%%%%%%%%%%%%%%%%（求V ）
%     t1 = clock;
     G =(2.*h+eps).^expo; %这里就是距离
    if length( find( sum(G) == 0 ) ) > 0
       V = X'*G./(( ones(size(X, 2), 1)*(sum(G)+eps) ));
    else
       V = X'*G./(( ones(size(X, 2), 1)*sum(G) ));  
    end
     V = V';   %转为行向量  cluster_n*dim_n
%     disp(['V:',num2str(etime(clock,t1))]);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%（求P ）
%     t1= clock;
%     H = EuDist2((X*P),(V*P));   %这里是距离，已经开平方,hik，V已经更新
%     G = 2.*h+eps;              %利用新的U     
  
    G1 = diag( sum(G,2) );
    G2 = diag( sum(G,1) ); 
    
    Q = EuDist2( X*P );     
    Ls0 = W./(2.*Q+eps);    
        
%      disp(['Ls0:',num2str(etime(clock,t1))]);
%      t1=clock;
    
    
    for i=1:1:data_n
        indx = find( Q(i,:) == 0 );
        Ls0(i,indx) = 0;
    end
    
    Ls1 = diag( sum(Ls0'+Ls0)/2 );
    Ls = Ls1 - (Ls0'+Ls0)/2;
%     disp(['Ls:',num2str(etime(clock,t1))]);
    
    D = sqrt( sum( P.^2, 2 ) );
    D = diag( 1./ ( 2 .* D +eps ) );
    
%     E = X'*G1*X - (X'*G*V + V'*G'*X) + V'*G2*V + Pmu*X'*Ls*X + Penta*D;
    E = (X'*G1*X/2+(X'*G1*X)'/2) - ((X'*G*V/2+ V'*G'*X/2)+(X'*G*V/2 + V'*G'*X/2)') +(V'*G2*V/2+(V'*G2*V)'/2) + Pmu*(X'*Ls*X/2+(X'*Ls*X)'/2)+ Penta*D;
    [eigvector eigvalue] = eig(E);    
%     eigvalue=abs(eigvalue);%复数求实数
%     eigvector=abs(eigvector);
    eigvalue = diag(eigvalue);            %%从小到大排列
    [junk, index] = sort(eigvalue);       %升序
    eigvalue = eigvalue(index);
    eigvector = eigvector(:, index);
    
%     maxEigValue = max(abs(eigvalue));
%     eigIdx = find(abs(eigvalue)/maxEigValue < 1e-12);
%     eigvalue (eigIdx) = [];
%     eigvector (:,eigIdx) = [];
    
    if ProK < length(eigvalue)             %取最小的k个特征值对应的特征向量
       eigvalue = eigvalue(1:ProK);
       eigvector = eigvector(:, 1:ProK);
    end
    P = eigvector;
    
    nsmp=size(P,2);   %dim_n*ProK  行向量
    for i=1:nsmp
       P(:,i)=P(:,i)/norm(P(:,i),2);
    end   
% [P, ps] = mapminmax(P', 0.1, 1);
% P=P';
    
    %%%% 计算obj
    obj(steps,1) = trace(P'*X'*G1*X*P)-2*trace(P'*X'*G*V*P) + trace(P'*V'*G2*V*P);
%     obj(steps,2) = Pgamma*norm(U,2);      %这里应该为F范数 z：这里注释了
    obj(steps,3) = Pmu*trace(P'*X'*Ls*X*P);
    obj(steps,4) = Penta*trace(P'*D*P);     
    obj(steps,5) = obj(steps,1)+obj(steps,3)+obj(steps,4);%z：把+obj(steps,2)删除
%       obj = 0;
    
    %if convergent?
    nsmp=size(V,1);   
    for i=1:nsmp
       ErrorV(i) = norm( V(i,:)-V_Old(i,:), 2);
    end 
    criterion = max( ErrorV );
    if criterion < conCriterion
        converged=true;
    end     
end 

%%--------------------end--------------------------------------------------