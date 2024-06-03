function v = lagrange(x, y, u)
n = length(x);
v = zeros(size(u));  % initialize the result vector so that the program may run faster.
for k = 1: n
    w = ones(size(u));
    for j = [1: k-1 k+1: n]
        w = (u-x(j))./(x(k)-x(j)).*w;
    end
    v = v+w*y(k);
end