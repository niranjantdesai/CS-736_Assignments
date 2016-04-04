    function [ c ] = classMeans( u,y,w,b,q,K )
%classMeans Finds the optimal value of class means within every iteration

c = zeros(K,1);
innerSumNum = conv2(b,w,'same');
innerSumDenom = conv2(b.^2,w,'same');

for k = 1:K
    num = sum(sum((u(:,:,k).^q).*y.*innerSumNum));
    denom = sum(sum((u(:,:,k).^q).*innerSumDenom));
    c(k) = num/denom;
end

end

