function [ J ] = objEval( y,mask,windowRadius,w,c,b,u,q,K )
%objEval Evaluates the objective function of modified FCM with bias
%correction 

imgSize = size(y);
d = zeros(K,1);
J=0;

for i=1:imgSize(1)
   for j=1:imgSize(2)
       if(mask(i,j) > 0)
           for k=1:K
               for l=-windowRadius:windowRadius
                  for m=-windowRadius:windowRadius
                      if(mask(i+l,j+m) > 0)
                         d(k) = d(k) + w(l+windowRadius+1, m+windowRadius+1)*((y(i,j) - c(k)*b(i+l,j+m))^2); 
                      end
                  end
               end
               J = J + (u(i,j,k)^q)*d(k);
           end
       end
   end
end

end

