% John driscoll sept 13th
% rectangular Chebyshev Space-time method for time-dependent heat with
% GMRES

% u_t - u_xx = 0

%f(-1) = f(1) = 0

Nx = 50; % Choose eps and N.
Nt = 50;


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
Ix = eye(Nx+2);
It = eye(Nt+1);

u0 = exp(-20*xFunc.^2) - exp(-20);

A = kron(It,-Dxx) + kron(Dt,Ix);
P = kron(Pt,Px);
% PA = P*A;
PA = kron(Pt,-Px*Dxx) + kron(Pt*Dt,Px);


exL = zeros(1,Nx+2);
exL(1) = 1;

exR = zeros(1,Nx+2);
exR(end) = 1;

% Boundary conditions at every time node
Bleft  = kron(It,exL);
Bright = kron(It,exR);

% Initial condition at interior spatial nodes only
Ix_interior = Ix(2:end-1,:);
Binit = kron([1 zeros(1,Nt)],Ix_interior);

B = [Bleft;
     Bright;
     Binit;
     PA];

rhs = [zeros(Nt+1,1);
       zeros(Nt+1,1);
       u0(2:end-1);
       zeros(size(PA,1),1)];


k = 5;

Px_low = sparse_barymat(xDE,xFunc,BaryW_x,k);
Pt_low = sparse_barymat(tDE,tFunc,BaryW_t,k);

P_low = kron(Pt_low,Px_low);

fprintf('||P-P_low||/||P|| = %.3e\n', ...
    norm(P-P_low,'fro')/norm(P,'fro'));

fprintf('||PA-P_low*A||/||PA|| = %.3e\n', ...
    norm(PA-P_low*A,'fro')/norm(PA,'fro'));

Bad_B = [Bleft;
     Bright;
     Binit;
     P_low*A];

setup.type = 'ilutp';
setup.droptol = 1e-3;
setup.udiag = true;


[Lbad,Ubad,Pbad] = lu(sparse(Bad_B));

prec = @(r) Ubad \ (Lbad \ (Pbad*r));

[x,flag,relres,iter,resvec] = gmres( ...
    B,rhs,[],1e-12,400,prec);

fprintf('\n--- GMRES with exact Bad_B inverse ---\n');
fprintf('Flag:              %d\n',flag);
fprintf('Preconditioned:    %.6e\n',relres);
fprintf('Original residual: %.6e\n',norm(B*x-rhs)/norm(rhs));
fprintf('Iterations:        %d\n',iter(2));


u = x;
u = reshape(u,[Nx+2,Nt+1]);

surf(u)

function P = sparse_barymat(Y,X,W,k)
    %just interpolate with close points

    M = length(Y);
    N = length(X);

    P = sparse(M,N);

    for i = 1:M

        [d,idx] = mink(abs(X-Y(i)),k);

        % Exact node match
        if d(1) < 10*eps
            P(i,idx(1)) = 1;
            continue
        end

        % Local barycentric interpolation
        vals = W(idx)./(Y(i)-X(idx));
        vals = vals/sum(vals);

        P(i,idx) = vals;

    end

end