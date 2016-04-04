function [ u ] = memberships( w,y,c,b,mask,K,q )
%memberships Finds optimal values of memberships within every iteration

imgSize = size(y);
u = zeros(imgSize(1),imgSize(2),K);
d = zeros(imgSize(1),imgSize(2),K);

maskSum = sum(sum(w));

t1 = conv2(b,w,'same');
t2 = conv2(b.^2,w,'same');
for i=1:K
    d(:,:,i) = ((y.^2).*maskSum - 2.*c(i).*y.*t1 + (c(i)^2).*t2);
end
d(d<0)=0;
u = (1./d).^(1/(q-1));
sum_u = nansum(u,3);

for i=1:K
    temp = u(:,:,i);
    temp = temp./sum_u;
    temp(~logical(mask))=0;
    u(:,:,i) = temp; 
end

u(find(isnan(u)))=0;

