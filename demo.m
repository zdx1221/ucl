%% Demonstration: first, generate synthetic model and data

D = 10;     % Number of common causes Z
rho = 0.5;  % Correlation of the common causes
v_x = 1;    % Variance of error term in the equation for treatment X
v_y = 1;    % Variance of error term in the equation for treatment Y
N   = 1000; % Sample size of 

% See the documentation of 'generate_model_data' to understand the
% structure of variables 'model' and 'dat'. The model has zero
% mean, although the empirical mean might be different from zero 
% (not by much).

[model, dat] = generate_model_data(D, rho, v_x, v_y, N);
figure(10)
scatter(dat.X,dat.Y)
title('Scatter plot between X and Y')
xlabel('X'); ylabel('Y')

%csvwrite('YXZ.csv',[dat.Y, dat.X, dat.Z]);
% data=csvread('non_YXZ_L.csv');
% fx=csvread('non_fx_L.csv');
% dat.Y=data(:,1);
% dat.X=data(:,2);
% dat.Z=data(:,3:end);
% [N,D]=size(dat.Z);



%% Plot true expected value E[Y | do(X = x)] for a range of values. 

% In our simulations, X always falls between 0 and 1, so the default range
% of values is between 0 and 1.

x_range = (0:0.01:1)';
fx = eval_do_x(model, dat, x_range);
%csvwrite('Ey.csv',fx);

% figure
% plot(x_range,fx)
% title('Expectation of outcome when varying treatment, comparison')
% xlabel('x'); ylabel('E[Y | do(X = x)]')

% The following is useful to check whether we are generating data in a
% reasonably large fraction of the [0, 1] space.

% x_sorted = sort(dat.X);
% fx_at_data = eval_do_x(model, dat, x_sorted);
% figure(1)
% plot(x_sorted, fx_at_data)
% title('Expectation of outcome when varying treatment at training points')
% xlabel('X'); ylabel('E[Y | do(X)]')

% The following generates a naive linear estimator without using the common
% causes. It is naive because it doesn't take into account the confounders
% X and the non-linearity. For simplicity, we assume we know the data are
% zero mean.

figure(2)
plot(x_range, fx, 'b')
title('Expectation of outcome when varying treatment, comparison 1')
xlabel('X'); ylabel('E[Y | do(X)]')
hold on


%beta_hat_1 =([dat.X,ones(N,1)]'*[dat.X,ones(N,1)]) \ ([dat.X,ones(N,1)]'*dat.Y);
%plot(x_range, [x_range , ones(length(x_range),1)] * beta_hat_1  , 'r')

% C0 = [ones(length(dat.Y), 1) dat.X dat.Y]; C = C0' * C0;
% beta_hat_1s = C(1:2, 1:2) \ C(1:2, 3);
% plot(x_range, x_range * beta_hat_1s(2) + beta_hat_1s(1), 'r')

% The following generates a naive linear estimator *using* the common
% causes. It is naive because it doesn't take into account the 
% non-linearity. For simplicity, we assume we know the data are
% zero mean.

%beta_hat_2 =([dat.X,ones(N,1),dat.Z]'*[dat.X,ones(N,1),dat.Z]) \ ([dat.X,ones(N,1),dat.Z]'*dat.Y);
%plot(x_range, [x_range , ones(length(x_range),1)] * beta_hat_2(1:2)  , 'g')
% C = cov([dat.X, dat.Z, dat.Y]);
% beta_hat_2 = C(1:(K + 1), 1:(K + 1)) \ C(1:(K + 1), end);
% plot(x_range, x_range * beta_hat_2(1), 'g')

%beta_hat_3 = ([dat.X, dat.X.^2, dat.X.^3, ones(N,1)]' * [dat.X, dat.X.^2, dat.X.^3, ones(N,1)])\([dat.X, dat.X.^2, dat.X.^3, ones(N,1)]'*dat.Y);
%plot(x_range, [x_range , x_range.^2, x_range.^3, ones(length(x_range),1)] * beta_hat_3  , 'r')
hold off
legend('True causal effect','Model 1')

%% Density estimation for X
x_range = (0:0.01:1)';
x_sorted = sort(dat.X);
x_inf_min = x_sorted(floor(N*0.025));
x_inf_max = x_sorted(floor(N*0.975));
x_inf = x_range(x_range>x_inf_min & x_range<x_inf_max);
X_density = ksdensity(dat.X, x_inf);
figure(11);
plot(x_range, ksdensity(dat.X, x_range))
xlabel('x')
ylabel('P(x)')
title('kernel smoothing density estimation')

%% X to Y regression

% hidden_size=10;
% net=feedforwardnet(hidden_size);
% view(net)
% net=train(net,x_range', fx');
% y_hat=net(x_range');
% figure;
% plot(x_range,y_hat')

    GPiters=300;
    M=50; % inducing points
    optionsGP = gpOptions('ftc');
    %optionsGP.numActive = M;   
    modelGPfitc = gpCreate(1, 1, x_range, fx, optionsGP);
    modelGPfitc = gpOptimise(modelGPfitc, 1, GPiters);
    [y_hat, ] = gpPosteriorMeanVar(modelGPfitc, x_range);
    figure;
    plot(x_range,y_hat)

% ind=randperm(101);
% model = Gaussian_Process_model('kernel',0.01,x_range(ind(1:20))',fx(ind(1:20))');
% [~,~,prediction] = Gaussian_Process(model,x_range' );
% y_hat=prediction(1,:);
% figure;
% plot(x_range',y_hat,'r')
% hold on
% plot(x_range,fx,'b')

%%  TODO: similar visual comparison, using plain Gaussian processes

% You might need to generate values for Z by Monte Carlo simulation and average
% over them numerically. The following does the linear model case again,
% but now with Monte Carlo simulation. beta_hat_2 is the model, but we will
% pretend we don't know that the causal effect is beta_hat_2(1). Instead,
% we evaluate E[Y | X, Z] using Z from the training set and swiping through
% the values of X.

fx_hat_gp = zeros(length(x_range), 1);


fprintf('# ----- Training a fitc GP... \n')
    GPiters=200;
    M=100; % inducing points
    optionsGP = gpOptions('fitc');
    optionsGP.numActive = M;   
    modelGPfitc = gpCreate(size(dat.X,2)+size(dat.Z,2), size(dat.Y,2), [dat.X,dat.Z], dat.Y, optionsGP);
    modelGPfitc = gpOptimise(modelGPfitc, 1, GPiters);  
for i = 1:length(x_range)
  % Evaluate E[Y] at all training configurations of Z, for a fixed value 
  % of X
  x = x_range(i);
  [muGPfitc, ] = gpPosteriorMeanVar(modelGPfitc, [x*ones(N,1),dat.Z]);
  fx_hat_gp(i) = mean(muGPfitc);  
  % You will need to substitute 'beta_hat_2(1) * x + dat.Z * beta_hat_2(2:end)' 
  % with whatever prediction the Gaussian process (or other black box 
  % model) generates.
end

figure(3)
plot(x_range, fx,'b')
title('Expectation of outcome when varying treatment, comparison 4')
xlabel('x'); ylabel('E[Y | do(X = x)]')
hold on
plot(x_range, fx_hat_gp, 'r') % This plot should be nearly identical to the previous one
hold on
%plot(x_inf, fx_hat_rec, 'k') 
legend('True causal effect','GP-fitc','MLP','Reconstruction')

%% neural network

fx_hat_mlp = zeros(length(x_range), 1);
cut = randperm(N);
hidden_to_err = [];
X = [dat.X,dat.Z];
D = size(X,2);
Y = dat.Y;

% for hidden_num=floor(D/2.):20
%     % Build a classifier model
%     net=feedforwardnet(hidden_num);
%     temp = cut;
%     valid_err=[];
%     for fold=1:5
%         temp=circshift(temp,[0,(fold-1)*floor(0.2*N)]);
%         ind_train=temp(1:floor(0.8*N));
%         ind_valid=temp(floor(0.8*N):end);
%         train_X = X(ind_train,:);
%         train_Y = Y(ind_train,:);
%         valid_X = X(ind_valid,:);
%         valid_Y = Y(ind_valid,:);
% 
%         % Train the model using LM
%         net=train(net,train_X', train_Y');
%         err=sum((net(valid_X')-valid_Y').^2);        
%         valid_err=[valid_err,err];
%     end
%     hidden_to_err=[hidden_to_err, mean(valid_err)];
% end
% [~ ,hidden_num_opt]=min(hidden_to_err);
% hidden_num_opt=hidden_num_opt+floor(D/2.)-1;

hidden_num_opt=13;
net=feedforwardnet(13);
%net.layers{2}.transferFcn = 'my_sin';

net=train(net,X', Y');
for i = 1:length(x_range)
  % Evaluate E[Y] at all training configurations of Z, for a fixed value 
  % of X
  x = x_range(i);
  fx_hat_mlp(i) = mean(net([x*ones(N,1),dat.Z]'));  
end
figure(4)
plot(x_range, fx,'b')
title('Expectation of outcome when varying treatment using MLP')
xlabel('x'); ylabel('E[Y | do(X = x)]')
hold on
plot(x_range, fx_hat_mlp', 'r') 
legend('True causal effect','MLP')

%%  Model of P(y|x) using MLP

net = feedforwardnet(10);
net = train(net,dat.X', dat.Y');
f_xy = net(x_inf');  
figure(4)
plot(x_inf, f_xy,'b')
title('E[Y|X] using MLP')
xlabel('x'); ylabel('E[Y | (X=x)]')

%%  Model of P(y|x) using GP

fprintf('# ----- Training a fitc GP... \n')
GPiters=200;
M=100; % inducing points
optionsGP = gpOptions('ftc');
optionsGP.numActive = M;
modelGP_yx = gpCreate(size(dat.X,2), size(dat.Y,2), dat.X, dat.Y, optionsGP);
modelGP_yx = gpOptimise(modelGP_yx, 1, GPiters);
[muGP_yx, ] = gpPosteriorMeanVar(modelGP_yx, x_inf);
figure(13);
plot(x_inf, muGP_yx)
title('Estimating E(Y|X) with GP')
xlabel('x')
ylabel('E[Y | (X=x)]')

%% Model of P(z|x) using FA
D = size(dat.Z, 2);
latent_dim = floor(D/2)+1;
sample_size = 200; % no more than 200
t_sample = {};
Joint = [dat.X, dat.Z];
mean_joint = mean(Joint);
std_joint = std(Joint);
Scaled_Joint = (Joint - repmat(mean_joint, N, 1)) ./ repmat(std_joint, N, 1);
[lamda, psi] = factoran(Scaled_Joint, latent_dim);
Cov = lamda*lamda' + diag(psi);
A = Cov(1,1);
B = Cov(2:end,2:end);
C = Cov(1,2:end);

for i = 1:length(x_inf)
    x = x_inf(i);
    scaled_x = (x - mean_joint(1)) / std_joint(1);
    mean_z_given_x = C/A*scaled_x;
    var_z_given_x = B - C'/A*C;
    t_sample{i} = randn(sample_size, D)*chol(var_z_given_x + 1e-6*eye(D)) + repmat(mean_z_given_x, sample_size, 1);
    t_sample{i} = t_sample{i}.*repmat(std_joint(2:end), sample_size, 1) + repmat(mean_joint(2:end), sample_size, 1); % transform back to original scale
end

%% Model of P(z|x) using GP assuming z are independent

[N, D] = size(dat.Z);
fprintf('# ----- Training a fitc GP... \n')
GPiters=200;
M=100; % inducing points
optionsGP = gpOptions('fitc');
optionsGP.numActive = M;
residual = zeros(N,D);
mean_z_given_x = zeros(length(x_inf),D);
sample_size = 200; % no more than 200
z_sample = {};

for d = 1:D   
    modelGP_zx_d = gpCreate(size(dat.X,2), 1, dat.X, dat.Z(:, d), optionsGP);
    modelGP_zx_d = gpOptimise(modelGP_zx_d, 1, GPiters);
    [muGP_zx_d, ] = gpPosteriorMeanVar(modelGP_zx_d, dat.X);
    residual(:,d) = dat.Z(:, d) - muGP_zx_d;
    [mean_z_given_x(:,d), ] = gpPosteriorMeanVar(modelGP_zx_d, x_inf);
    %figure;
    %plot(x_range, mean_z_given_x(:,d));
    disp(d);
end

Cov_err = cov(residual);

for i = 1:length(x_inf)    
    z_sample{i} = randn(sample_size, D)*chol(Cov_err + 1e-6*eye(D)) + repmat(mean_z_given_x(i,:), sample_size, 1);
end
        

%% Optimization
X = [dat.X, dat.Z];
Y = dat.Y;
Z = t_sample;
Ey = muGP_yx;
L2 = 5e-3;

% w0 = [ones(1,11), 1, 0.001];
% options = optimset('GradObj','on');
% f = @(w)objfunc( w, X, Y, Z, Ey, x_range);
% [w, fval, exitflag, output] = fminunc(f, w0, options);


% eta = 0.01;
% w = [randn(1,12),0.01];
% sum_delta = ones(1, length(w));
% for i=1:20    
%     [f,g]=objfunc(w, X, Y, Z, Ey, x_range);
%     disp(f);
%     disp(i);
%     sum_delta = sum_delta + abs(g);
%     w = w - eta * g ./ sqrt(sum_delta);
% end

w_init = [ones(1,D+2), 0.1];
options = zeros(1,20);
options(1) = 1;
options(2) = 1e-4;
options(3) = 1e-6;
options(14) = 70; % iterations
[w_opt, options] = my_scg('objfunc', w_init, options, X, Y, Z, Ey, x_inf, X_density, L2);

[~,~, y_hat] = objfunc( w_opt, X, Y, Z, Ey, x_inf, X_density, L2);
figure;
plot(x_inf, y_hat,'r')
hold on
plot(x_inf, muGP_yx,'b')
title('differnce between the estimation of E(y|x)')
xlabel('x')
ylabel('E(y|x)')
legend('SUM_z[E(y|z,x)*P(z|x)]','E(y|x)')

%% Evaluate the optimized model

%w0 = [randn(1,12),0.01];
fx_hat_rec = [];
model_opt = GP_CreateModel('ardkernel', w_opt, [dat.X,dat.Z], dat.Y);

for i = 1:length(x_range)
    % Evaluate E[Y] at all training configurations of Z, for a fixed value
    % of X
    disp(i);
    x = x_range(i);
    [~,~,prediction] = GP_Posterior( model_opt ,[x*ones(N,1),dat.Z] );
    fx_hat_rec(i) = mean(prediction);
    % You will need to substitute 'beta_hat_2(1) * x + dat.Z * beta_hat_2(2:end)'
    % with whatever prediction the Gaussian process (or other black box
    % model) generates.
end

figure;
plot(x_range, fx,'b')
title('Estimation of causal effect through reconstruction')
xlabel('x'); ylabel('E[Y | do(X = x)]')
hold on
plot(x_range, fx_hat_rec, 'r') % This plot should be nearly identical to the previous one
legend('True causal effect','GPLVM')

%%  Model of P(x,z) using GP_LVM

% Fix seeds
randn('seed', 1e6);
rand('seed', 1e6);


% load data
Ytr = [ dat.X, dat.Z];

% Set up model
options = vargplvmOptions('dtcvar');
options.kern = 'rbfardjit';%{'rbfard2', 'white'};
options.numActive = 50; 
options.initSNR = 100;
options.initX = 'vargplvm';
%options.scale2var1 = 1; % scale data to have variance 1

options.optimiser = 'scg2';
latentDim = 10;

model_lvm = vargplvmCreate(latentDim, size(Ytr, 2), Ytr, options);
model_lvm = vargplvmParamInit(model_lvm, model_lvm.m, model_lvm.X, options); 
model_lvm.vardist.covars = 0.5*ones(size(model_lvm.vardist.covars)) + 0.001*randn(size(model_lvm.vardist.covars));

iters = 1000;
display = 1;

fprintf('# Initialising the variational distribution for 1000 iterations...\n')
model_lvm.initVardist = 1; model_lvm.learnSigmaf = 0; model_lvm.learnBeta = 0;
model_lvm = vargplvmOptimise(model_lvm, display, 500);
modelInit = model_lvm;

fprintf(['# Optimising the model for ' num2str(iters) ' iterations...\n'])
model_lvm.initVardist = 0; model_lvm.learnSigmaf = 1; model_lvm.learnBeta = 1;
model_lvm = vargplvmOptimise(model_lvm, display, iters);


%% PREDICTIONS (reconstruction of test outputs, P(z|x))

testIters = 100;

fprintf(['\n\n# Test phase: each point optimised for ' num2str(testIters) ' iterations...\n'])

% patrial reconstruction of test points

Yts = [x_inf, NaN*ones(length(x_inf), size(dat.Z,2))];
mini = zeros(1, size(Yts,1));
Varmu = zeros(size(Yts,1), size(Yts,2)-1);
Varsigma = {};
for i=1:size(Yts,1)
    fprintf(['\n\n# Test point # ' num2str(i) '...\n\n'])
    %
    % randomly choose which outputs are present
    
    indexPresent =  1; % x,z
    indexMissing = 2:size(dat.Z,2)+1; % y
    % initialize the latent point using the nearest neighbour
    % from he training data
    dst = dist2(Yts(i,indexPresent), Ytr(:,indexPresent));
    [mind, mini(i)] = min(dst);
    
    % create the variational distribtion for the test latent point
    vardistx = vardistCreate(model_lvm.vardist.means(mini(i),:), model_lvm.q, 'gaussian');
    vardistx.covars = 0.2*ones(size(vardistx.covars));
    
    % optimize mean and vars of the latent point
    model_lvm.vardistx = vardistx;
    [x, varx] = vargplvmOptimisePoint(model_lvm, vardistx, Yts(i, :), display, testIters); %
    
    % reconstruct the missing outputs
    [mu, sigma] = vargplvmMissingValuePredict(model_lvm, x, varx, indexMissing);
    Varmu(i,:) = mu;
    Varsigma{i} = sigma{1}; 
end

sample_size = 200;
D = size(dat.Z, 2);
for i = 1:length(x_inf)    
    lvm_sample{i} = randn(sample_size, D)*chol(Varsigma{i} + 1e-6*eye(D)) + repmat(Varmu(i,:), sample_size, 1);
end
