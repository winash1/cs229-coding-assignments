clear
x=load('logistic_x.txt');
x=[ones(length(x(:,1)),1) x];
x=x';
y=load('logistic_y.txt');
y=y';
theta=[0;0.2;0];
oldtheta=[1;1;1];
i=1;
while norm(oldtheta - theta) > 1e-15    
h_theta=1./(1+exp(-y'.*(x'*theta)));
gradient=(x*((1-h_theta).*y'));
hessian=(x*diag(h_theta.*(h_theta-1))*x');
oldtheta=theta;
theta=theta-(hessian^-1)*gradient;
end
figure;
gscatter(x(2,:),x(3,:),y,'br','xo');
hold on
tspan=min(x(2,:)):max(x(2,:));
plot(tspan,-(tspan.*theta(2)+theta(1))./theta(3));

    
