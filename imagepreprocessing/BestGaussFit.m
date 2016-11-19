function [ p ] = BestGaussFit( t,y,p,residualThreshold )

iter = 1;
N = length(y);

dg_da = zeros(N,1);
dg_dmu = zeros(N,1);
dg_dsigma = zeros(N,1);
g = zeros(N,1);

while true
    
    for i = 1:N
        dg_da(i) = exp(-(t(i)-p(2))^2/(p(3)^2));
        dg_dmu(i) = p(1)*(2*(t(i)-p(2))/(p(3)^2))*exp(-(t(i)-p(2))^2/(p(3)^2));
        dg_dsigma(i) = p(1)*(2*(t(i)-p(2))^2/(p(3)^3))*exp(-(t(i)-p(2))^2/(p(3)^2));
        g(i) = p(1)*exp(-(t(i)-p(2))^2/(p(3)^2));
    end
    
    A = [dg_da dg_dmu dg_dsigma];
    
    Y = y-g;
    
    xls = pinv(A)*Y;
    
    p = p+xls;
    
    RMS_error(iter) = sqrt((1/N)*(A*xls - Y)'*(A*xls - Y));
    
    if iter ~= 1
        deltaE = (RMS_error(iter) - RMS_error(iter-1))/RMS_error(iter);
    else
        deltaE = RMS_error(iter)/RMS_error(iter);
    end
    
    if abs(deltaE)<residualThreshold
        break
    end
    
    iter = iter+1;
    
end

figure();

plot(1:iter,RMS_error)
xlabel('Iterations');
ylabel('RMS Error');
title('Nonlinear Appriximation Error Convergence');

figure()
    
gauss_fit = p(1)*exp(-(t-p(2)).^2/(p(3)^2));
plot(t,gauss_fit)
hold on
scatter(t,y)
xlabel('Time');
ylabel('Gaussian Fit');
title('Gaussian Fit vs Data');
legend('Gaussian Fit','Sample Data');

end

