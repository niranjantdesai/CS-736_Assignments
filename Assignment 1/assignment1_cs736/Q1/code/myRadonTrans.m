function [t,theta,out_data] = myRadonTrans( input_data,delS )
%myRadonTrans Performs radon transformed along the predecided (t,theta) values

t = -90:5:90;
theta = 0:5:175;

argMat = zeros(length(t),length(theta),2);

argMat(:,:,1) = repmat(t',1,length(theta));
argMat(:,:,2) = repmat(theta/(2*pi),length(t),1);

argMat = reshape(argMat,[],2);

% creating argument matrices to use arrayfun

handle = @(t,theta) myIntegration(input_data,t,theta,delS);
out_data = arrayfun(handle,argMat(:,1),argMat(:,2));

out_data = reshape(out_data,length(t),length(theta));

end

