% John driscoll sept 13th
% rectangular Chebyshev Space-time method for time-dependent heat condition
% number




Nx_vals = 10:10:50;
Nt_vals = 10:10:100;

COND_MATRIX = zeros([length(Nx_vals),length(Nt_vals)]);

progress = 0;
total = length(Nx_vals)*length(Nt_vals);

for x_iter = 1:length(Nx_vals)
    for t_iter = 1:length(Nt_vals)
        progress = progress +1;
        progress/total


        Nx = Nx_vals(x_iter);
        Nt = Nt_vals(t_iter);
        
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

        COND_MATRIX(x_iter,t_iter) = cond(B);
    end
end


imagesc(Nx_vals, Nt_vals, COND_MATRIX);
colorbar;
xlabel('Nx values');
ylabel('Nt values');
title('Condition Number Heatmap');