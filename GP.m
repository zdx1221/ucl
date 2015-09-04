%% linear regression
load co2.txt
Y=co2(:,3)';
t=(co2(:,1)+(co2(:,2)-1)/12)';
figure;
plot(t,Y)
title('global CO2 concentrations')
xlabel('time')
ylabel('parts per million')
sigma2=1;                              % observation noise
lamda=[0;360];                         % prior mean
C=[100,0;0,10000];                     % prior covariance
X=[t;ones(1,size(t,2))];
cov=((X*X'/sigma2+C\eye(2)))\eye(2);   % posterior covariance
u=cov*(X*Y'/sigma2+C\lamda);           % posterior mean
g_obs=Y-u'*X;                          % residual
figure;
plot(t,g_obs)
title('residual')
xlabel('time')
g=reshape(g_obs(1:336),12,28);
g12=g-repmat(g(1,:),12,1);             % variation in each year
figure;
plot(mean(g))                          % annual mean of residual
title('annual mean of residual')
xlabel('year')
figure;
plot(g12)     
title('variation of the residual in each year')
xlabel('month')

%% draw samples from GP
xi2=0.02;                              % observation noise
[prior,posterior] = Gaussian_Process('kernel',xi2,t,[g_obs;1:344] );
figure;
plot(prior');
title('prior of 3 random samples')
figure;
h=area([posterior(4,:)'-2*posterior(5,:)',4*posterior(5,:)'],-5);
hold on
set(h(1),'FaceColor',[1,1,1]);
set(h(2),'FaceColor',[0.85,0.85,0.85]);% gray area represents 95% confidence interval
set(h,'EdgeColor',[0.85,0.85,0.85]);
plot(posterior(1:3,:)');
title('posterior of 3 random samples above')

%% hyperparameter selection
xi2=0.02;                               % observation noise                     
i=1:30:301;                             % 10 traning samples
gt=[g_obs(i);i];
[~,posterior] = Gaussian_Process('kernel',xi2,t,gt);
figure;
h=area([posterior(4,:)'-2*posterior(5,:)',4*posterior(5,:)'],-5);
hold on
set(h(1),'FaceColor',[1,1,1]);
set(h(2),'FaceColor',[0.85,0.85,0.85]); % gray area represents 95% confidence interval
set(h,'EdgeColor',[0.85,0.85,0.85]);
plot(posterior(1:3,:)');
hold on
plot(i,gt(1,:),'xk','MarkerSize',10)    % training points with crossing mark
title('posterior with 10 training points')

%% prediction
num=1:(4+13*12); 
t_pre=2007+(8-1)/12+num/12;
[~,~,pre] = Gaussian_Process('kernel',xi2,t,g_obs,t_pre );
prediction=pre(1,:)+u'*[t_pre;ones(1,size(t_pre,2))];
figure;
plot([t,t_pre],[Y,prediction]);
hold on
h=area(repmat(t_pre',1,2),[prediction'-2*pre(2,:)',4*pre(2,:)']);
hold on
set(h(1),'FaceColor',[1,1,1]);
set(h(2),'FaceColor',[0.80,0.80,0.80]);
set(h(1),'EdgeColor',[1,1,1]);
set(h(2),'EdgeColor',[0.80,0.80,0.80]);
plot(t_pre,prediction,'-k');
legend('observation','prediction interval','95% confidence interval','prediction mean');
title('Prediction of CO2 concentrations from 2007 to 2020')
ylabel('parts per million')
xlabel('time')
axis([1975,2025,300,450]);

%% modelling f(t) with Gaussian Process
[~,~,pre] = Gaussian_Process('kernel2',xi2,t,Y,t_pre );
prediction=pre(1,:);
figure;
plot([t,t_pre],[Y,prediction]);
hold on
h=area(repmat(t_pre',1,2),[prediction'-2*pre(2,:)',4*pre(2,:)']);
hold on
set(h(1),'FaceColor',[1,1,1]);
set(h(2),'FaceColor',[0.80,0.80,0.80]);
set(h(1),'EdgeColor',[1,1,1]);
set(h(2),'EdgeColor',[0.80,0.80,0.80]);
plot(t_pre,prediction,'-k');
legend('observation','prediction interval','95% confidence interval','prediction mean');
title('Prediction of CO2 concentrations from 2007 to 2020 with GP')
ylabel('parts per million')
xlabel('time')
axis([1975,2025,300,450]);

