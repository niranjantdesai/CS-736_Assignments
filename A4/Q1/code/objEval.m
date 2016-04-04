function [ J ] = objEval( y,w,c,b,u,q,K )
%objEval Evaluates the objective function of modified FCM with bias
%correction 

imgSize = size(y);
d = zeros(imgSize(1),imgSize(2),K);
maskSum = sum(sum(w));

t1 = conv2(b,w,'same');
t2 = conv2(b.^2,w,'same');
t3 = zeros(imgSize);
for i=1:K
    d(:,:,i) = (y.^2).*maskSum - 2.*c(i).*y.*t1 + (c(i)^2).*t2;
end

for k=1:K
    t3 = t3 + (u(:,:,k).^q).*d(:,:,k);
end

J = sum(sum(conv2(w,t3,'same')));

end

