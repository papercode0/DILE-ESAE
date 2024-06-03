function [P V steps obj] = RJSFCM(X,W,dataY1, ProK,Pmu,Penta,maxStep,conCriterion)
%% robust jointly sparse fuzzy C-means
% X:���ݼ��� n*m����������ʽ
% W:��LLE���������ƶȾ���
% ProK��ѡ���������Ŀ
%Pgamma��Pmu, Penta������
% maxStep������������
% conCriterion ����������
%P: m*ProK;
%V cluster_n*m  ������
%U n*cluster_n  ������
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

P = eye(dim_n,ProK);    %��ʼ��Ϊ��λ��

steps=0;
converged=false;

while ~converged && steps<=maxStep
     
    steps=steps+1; 
    
    V_Old = V;
    
     %%%%%%%%%%%%%%%%%%%%%����Uik��  %z:��ֵ����û���õ�������ע�����������
%      t1 = clock;
       %%%% ����hik��||XiP-VkP||_2  
      h = EuDist2((X*P),(V*P));   %%���룬�Ѿ�������L21
%  
%      for i=1:1:data_n
%         vi = -1*h(i,:)/(2*Pgamma);
%         U(i,:) = EProjSimplex(vi);
%      end     
%      disp(['U:',num2str(etime(clock,t1))]);
     
    %%%%%%%%%%%%%%%%%%%%%%%%%%����V ��
%     t1 = clock;
     G =(2.*h+eps).^expo; %������Ǿ���
    if length( find( sum(G) == 0 ) ) > 0
       V = X'*G./(( ones(size(X, 2), 1)*(sum(G)+eps) ));
    else
       V = X'*G./(( ones(size(X, 2), 1)*sum(G) ));  
    end
     V = V';   %תΪ������  cluster_n*dim_n
%     disp(['V:',num2str(etime(clock,t1))]);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%����P ��
%     t1= clock;
%     H = EuDist2((X*P),(V*P));   %�����Ǿ��룬�Ѿ���ƽ��,hik��V�Ѿ�����
%     G = 2.*h+eps;              %�����µ�U     
  
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
%     eigvalue=abs(eigvalue);%������ʵ��
%     eigvector=abs(eigvector);
    eigvalue = diag(eigvalue);            %%��С��������
    [junk, index] = sort(eigvalue);       %����
    eigvalue = eigvalue(index);
    eigvector = eigvector(:, index);
    
%     maxEigValue = max(abs(eigvalue));
%     eigIdx = find(abs(eigvalue)/maxEigValue < 1e-12);
%     eigvalue (eigIdx) = [];
%     eigvector (:,eigIdx) = [];
    
    if ProK < length(eigvalue)             %ȡ��С��k������ֵ��Ӧ����������
       eigvalue = eigvalue(1:ProK);
       eigvector = eigvector(:, 1:ProK);
    end
    P = eigvector;
    
    nsmp=size(P,2);   %dim_n*ProK  ������
    for i=1:nsmp
       P(:,i)=P(:,i)/norm(P(:,i),2);
    end   
% [P, ps] = mapminmax(P', 0.1, 1);
% P=P';
    
    %%%% ����obj
    obj(steps,1) = trace(P'*X'*G1*X*P)-2*trace(P'*X'*G*V*P) + trace(P'*V'*G2*V*P);
%     obj(steps,2) = Pgamma*norm(U,2);      %����Ӧ��ΪF���� z������ע����
    obj(steps,3) = Pmu*trace(P'*X'*Ls*X*P);
    obj(steps,4) = Penta*trace(P'*D*P);     
    obj(steps,5) = obj(steps,1)+obj(steps,3)+obj(steps,4);%z����+obj(steps,2)ɾ��
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