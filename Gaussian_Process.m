function [prior,posterior,prediction] = Gaussian_Posterior( model ,X_pre )
% Gaussian_Process: providing prior,posterior and prediction with given
% training data and kernel according to gaussian process
% input:
% kernel:     a string of kernel name
% xi2:        observation noise
% X:          training input
% Y:          observed value(neefd to add a vector of index when computing posterior)
% X_pre:      prediction input
% output:
% prior:      prior of random sample
% posterior:  posterior of random sample(1st row), posterior mean(2nd row), standard deviation(3rd row)
% prediction: prediction mean(1st row), standard deviation(2nd row)

% computing prediction
if nargin>1
    prior=[];posterior=[];
    K_pre_X=feval(model.kernel, X_pre, model.X, model);
    ave=K_pre_X * model.inverse_K_Y;
    %std_err=sqrt(diag(feval(model.kernel, X_pre, X_pre, model.xi2) - K_pre_X / K * K_pre_X'));
    prediction=ave';%;std_err'];
% computing posterior and prior
% elseif nargin>3
%     [~,n]=size(X);prediction=[];
%     K=feval(kernel,X,X);
%     K=(K+K')/2;
%     L=chol(K+0.0001*eye(n));                    % avoid numerical problem
%     random=randn(3,n);
%     prior=random*L;
%     x=X(Y(2,:));                                % training input corresponding to observed value
%     K_noise=feval(kernel,x,x,xi2);
%     K_noise=(K_noise+K_noise')/2;
%     K_Xx=feval(kernel,X,x);
%     L=chol(K-K_Xx/K_noise*K_Xx'+0.0001*eye(n)); % avoid numerical problem
%     ave=K_Xx/K_noise*Y(1,:)';
%     std_err=sqrt(diag(K-K_Xx/K_noise*K_Xx'));
%     posterior=[random*L+repmat(ave',3,1);ave';std_err'];
% % computing prior
% else
%     [~,n]=size(X);posterior=[];prediction=[];
%     K=feval(kernel,X,X);
%     K=(K+K'/2);
%     L=chol(K+0.0001*eye(n));                    % avoid numerical problem
%     prior=randn(3,n)*L;
% end   
end
