% John driscoll sept 13th
% rectangular Chebyshev Space-time method for time-dependent heat

% u_t - u_xx = 0

%f(-1) = f(1) = 0

Nx = 40; % Choose eps and N.
Nt = 55;


% Square, scaled differentiation matrix for 2nd-kind points:
Dx = diffmat(Nx+2); 
Dxx = Dx*Dx;

Dt = 2*diffmat(Nt+1); %N points plut one boundry condtion in time  we will by

% Nodes for solution, plus quadrature and barycentric weights:
[xFunc, CCW_x, BaryW_x] = chebpts(Nx+2, [-1 1], 2);
[tFunc, CCW_t, BaryW_t] = chebpts(Nt+1, [0 1], 2);

[XFunc,TFunc] = meshgrid(xFunc,tFunc);

xDE = chebpts(Nx, [-1, 1], 1); % 1st-kind grid for ODE.
tDE = chebpts(Nt, [0, 1], 1); % 1st-kind grid for ODE.

[XDE,TDE] = meshgrid(xDE,tDE);

Px = barymat(xDE, xFunc, BaryW_x); % Resampling matrix P*f(Func) = f(DE) 
Pt = barymat(tDE, tFunc, BaryW_t); % Resampling matrix P*f(Func) = f(DE) 

Ix = eye(Nx+2); % Identity operator.
It = eye(Nt+1);


u0 = exp(-20*xFunc.^2) - exp(-20);

A = kron(It, -Dxx) + kron((Dt), Ix);
PA = kron(Pt,Px)*A; 

exL = zeros(1,Nx+2);
exL(1) = 1;

exR = zeros(1,Nx+2);
exR(end) = 1;

Bleft  = kron(It,exL);
Bright = kron(It,exR);

Binit = kron([1 zeros(1,Nt)],Ix);

B = [Bleft;
     Bright;
     Binit;
     PA];

rhs = [zeros(Nt+1,1);
       zeros(Nt+1,1);
       u0(:);
       zeros(size(PA,1),1)]; % space left, space right, time start, pde

rhs = [zeros(Nt+1,1);
       sin(3*pi*tFunc).^2;
       u0(:);
       zeros(size(PA,1),1)]; % space left, space right, time start, pde




% Solve the linear system B * u = rhs for the initial condition
u = B \ rhs;

u = reshape(u,[Nx+2,Nt+1]);
surf(u)

