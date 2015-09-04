function model = GP_CreateModel( kernel,xi2,X,Y )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% initaial value
sigma2 = 1;
A = ones(1,size(X,2));
model.kernel_parameter{1} = sigma2;;
model.kernel_parameter{2} = A;

model.Y = Y;
model.X = X;
model.xi2 = xi2;
K = feval(kernel, X, X, model, xi2);
model.K = (K+K')/2;
model.kernel = kernel;
model.inverse_K_Y = model.K\Y;

end

