function [CKSym,CAbs] = BuildAdjacency(CMat,K)

if (nargin < 2)
    K = 0;
end

N = size(CMat,1);
CAbs = abs(CMat);

[Srt,Ind] = sort( CAbs,1,'descend' );

if (K == 0)
    for i = 1:N
        CAbs(:,i) = CAbs(:,i) ./ (CAbs(Ind(1,i),i)+eps);
    end
else
    for i = 1:N
        for j = 1:K
            CAbs(Ind(j,i),i) = CAbs(Ind(j,i),i) ./ (CAbs(Ind(1,i),i)+eps);
        end
    end
end

CKSym = CAbs + CAbs';