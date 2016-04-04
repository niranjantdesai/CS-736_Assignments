function [ u ] = memberships( w,y,c,b,mask,K,q )
%memberships Finds optimal values of memberships within every iteration

imgSize = size(y);
u = zeros(imgSize(1),imgSize(2),K);
d = zeros(imgSize(1),imgSize(3),K);

maskSum = sum(sum(w));

t1 = conv2(b,w);
t2 = conv2(b.^2,w);
for i=1:K
    d(:,:,i) = y.*maskSum - 2.*c(i).*y.*t1 + c(i).*t2;
end

u = (1./d).^(1/(q-1));

for i=1:K
    temp = u(:,:,i);
    temp(~logical(mask))=0;
    u(:,:,i) = temp;
end

