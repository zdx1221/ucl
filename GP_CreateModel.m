function model = GP_CreateModel( kernel, parameter, X, Y )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here



model.kernel_parameter{1} = parameter(end-1);
model.kernel_parameter{2} = parameter(1:end-2);
model.xi = parameter(end);
model.Y = Y;
model.X = X;

K = feval(kernel, X, X, model, model.xi);
model.K = (K+K')/2;
model.kernel = kernel;
model.inverse_K_Y = model.K\Y;

end

