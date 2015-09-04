function [mse, gradient, y_hat] = objfunc( w, X, Y, Z_sample, Ey, x_inf, X_density, L2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% w(end-1) : sigma
% w(end) : xi
% w(1:d) : ARD weight of each dimension
% X : training input
% Y : training output
% Z : samples from P(z|x)
% Ey : E(y|x)



[a,d] = size(X);
xx = zeros(a,a);
for j = 1:d
    xx_d{j} = (repmat(X(:,j),1,a) - repmat(X(:,j)',a,1)).^2;
    xx = xx + xx_d{j}*w(j)^2;
end
K_xx = w(end-1)^2*exp(-0.5*xx);

% for i =1:length(Z_sample)
%     Z = [x_range(i)*ones(size(Z_sample{i},1),1), Z_sample{i}]; 
%     [b,d] = size(Z);
%     zx = zeros(b,a);
%     for j = 1:d
%         zx_d{i,j} = (repmat(Z(:,j),1,a) - repmat(X(:,j)',b,1)).^2;
%         zx = zx + zx_d{i,j}*w(j)^2;
%     end
%     K_zx{i} = w(end-1)^2*exp(-0.5*zx);
%     y_hat(i) = mean(K_zx{i} / (K_xx + w(end)^2*eye(a)) * Y);
% end

b = size(Z_sample{1},1);
sum_g_Kzx = zeros(b,a);
sum_g_Kzx_d = cell(1,d);

for i =1:length(Z_sample)
    Z = [x_inf(i)*ones(size(Z_sample{i},1),1), Z_sample{i}]; 
    zx = zeros(b,a);
    for j = 1:d
        zx_d_i{j} = (repmat(Z(:,j),1,a) - repmat(X(:,j)',b,1)).^2;
        zx = zx + zx_d_i{j}*w(j)^2;
    end
    K_zx_i = w(end-1)^2*exp(-0.5*zx);
    y_hat(i) = mean(K_zx_i / (K_xx + w(end)^2*eye(a)) * Y);
    err(i) = y_hat(i) - Ey(i);
    sum_g_Kzx = sum_g_Kzx + K_zx_i * err(i) * X_density(i);
    
    for j = 1:d
        if i ==1
            sum_g_Kzx_d{j} = zeros(b,a);
        end
        sum_g_Kzx_d{j} = sum_g_Kzx_d{j} + K_zx_i.*zx_d_i{j} * w(j) * err(i) * X_density(i);
    end
    
 end 


e = ones(b,1)/b;
%err = y_hat' - Ey;
mse = 0.5* err * (X_density .* err') + L2 *0.5 * w(1:d) * w(1:d)';


% Compute Gradient

if nargout>1
    
    gradient = zeros(1, length(w));
    
%     for j = 1:d
%         sum_g_Kzx_d{j} = zeros(b,a);
%         for i = 1:length(Z_sample)
%             sum_g_Kzx_d{j} = sum_g_Kzx_d{j} + K_zx{i}.*zx_d{i,j} * err(i) * w(j);
%         end
%     end
%     
%     sum_g_Kzx = zeros(b,a);
%     for i = 1:length(Z_sample)
%         sum_g_Kzx = sum_g_Kzx + K_zx{i} * err(i);
%     end
    
    % gradient wrt ARD weight
    
    common_1 = (K_xx + w(end)^2*eye(a)) \ Y ;
    common_2 = sum_g_Kzx / (K_xx + w(end)^2*eye(a));
    for j =1:d
        gradient(j) =  e' * ( common_2 * (w(j)*K_xx.*xx_d{j}) - sum_g_Kzx_d{j} ) * common_1 + w(j)*L2;
    end
    
    % gradient wrt sigma
    
    gradient(end-1) = 2*e' * ( sum_g_Kzx / w(end-1) - common_2  * K_xx / w(end-1) ) * common_1;
    
    % gradient wrt xi
    
    gradient(end) = -2*e'* w(end) * common_2 * common_1 ;
    
    
end

end



