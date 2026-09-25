% rectangular Chebyshev Space-time method for time-dependent schrodinger equation

% i*u_t = -u_xx + V(x)u
% u_t = -i*(-u_xx + V(x)u)
%v(x) = 20(x^2-0.25)^2   %double well

%u(-1,t) = u(1,t) = 0

Nx = 45;
Nt = 120;


% Square, scaled differentiation matrix for 2nd-kind points:
Dx = diffmat(Nx+2); 
Dxx = Dx*Dx;

Dt = diffmat(Nt+1); %N points plut one boundry condtion in time  we will by

% Nodes for solution, plus quadrature and barycentric weights:
[xFunc, CCW_x, BaryW_x] = chebpts(Nx+2, [-1 1], 2);
[tFunc, CCW_t, BaryW_t] = chebpts(Nt+1, [0 2], 2);

[XFunc,TFunc] = meshgrid(xFunc,tFunc);

xDE = chebpts(Nx, [-1, 1], 1); % 1st-kind grid for ODE in x.
tDE = chebpts(Nt, [0, 2], 1); % 1st-kind grid for ODE in t.

[XDE,TDE] = meshgrid(xDE,tDE);

Px = barymat(xDE, xFunc, BaryW_x); % Resampling matrix P*f(Func) = f(DE) 
Pt = barymat(tDE, tFunc, BaryW_t); % Resampling matrix P*f(Func) = f(DE) 

Ix = eye(Nx+2); % Identity operator.
It = eye(Nt+1);

u0 = (1-xFunc.^2).*exp(-25*(xFunc+0.5).^2).*exp(1i*1*xFunc);

V = 400*(xFunc.^2 - 0.25).^2 + 50*(xFunc.^2 - 0.25).^2 + 100*exp(-80*xFunc.^2);
Vmat = diag(V);


%OPERATOR
A = kron(Dt,Ix) - 1i*kron(It,Dxx)  + 1i*kron(It,Vmat);

PA = kron(Pt,Px)*A; %is this right? 


exL = zeros(1,Nx+2);
exL(1) = 1;

exR = zeros(1,Nx+2);
exR(end) = 1;

% Boundary conditions for t > 0
timeBC = eye(Nt+1);
timeBC(1,:) = [];

Bleft  = kron(timeBC,exL);
Bright = kron(timeBC,exR);

% Initial condition at t = 0
Binit = kron([1 zeros(1,Nt)],Ix);

%BC's + PRojector Operator
B = [Bleft;
     Bright;
     Binit;
     PA];

rhs = [zeros(Nt,1);
       zeros(Nt,1);
       u0(:);
       zeros(size(PA,1),1)];

u = B\rhs;
u = reshape(u,[Nx+2,Nt+1]);

prob = abs(u).^2;

figure
surf(xFunc,tFunc,prob')
shading interp
xlabel('x')
ylabel('t')
zlabel('|u|^2')
title('Probability density')
view(45,30)



%CHATGPT CODE


mass = zeros(1,Nt+1);

for i = 1:Nt+1
    mass(i) = CCW_x * abs(u(:,i)).^2;
end

figure
plot(tFunc,mass,'LineWidth',2)
xlabel('t')
ylabel('\int |u|^2 dx')
grid on

fprintf('Initial mass: %.16e\n',mass(1))
fprintf('Final mass:   %.16e\n',mass(end))
fprintf('Relative error: %.3e\n', ...
    max(abs(mass-mass(1)))/mass(1))

fprintf('\nSelected time slices:\n')
fprintf('       t                 mass              relative error\n')

for i = round(linspace(1,Nt+1,6))
    fprintf('%8.4f    %.16e    %.3e\n', ...
        tFunc(i),mass(i),(mass(i)-mass(1))/mass(1));
end