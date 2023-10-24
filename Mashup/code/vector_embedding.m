function [w, x, P, fval] = vector_embedding(Q, ndim, maxiter)
    [nnode ncontext] = size(Q);
    nparam = (nnode + ncontext) * ndim;
    
    %% Optimize
    opts = struct('factr', 1e4, 'pgtol', 0, 'm', 5, 'printEvery', 50, 'maxIts', maxiter);

    while true
      %% Initialize vectors
      fprintf('Initializing vectors ... '); tic
      wx = rand(ndim, nnode + ncontext) / 10 - .05;
      fprintf('done. '); toc

      opts.x0 = wx(:);
      [xopt, fval, info] = lbfgsb(@optim_fn, -inf(nparam,1), inf(nparam,1), opts);
      if info.iterations > 10
        break
      end
      fprintf('Premature termination (took %d iter to converge); trying again.\n', info.iterations);
      info
    end
    wx = reshape(xopt, ndim, nnode + ncontext);

    fprintf('Done.\n');
    
    %% Summarize output
    w = wx(:,1:ncontext);
    x = wx(:,ncontext+1:end);
    P = P_fn(w,x);
    fval = obj_fn(P);
    
    function [fval, grad] = optim_fn(wx)
        wx = reshape(wx, ndim, nnode + ncontext);

        P = P_fn(wx(:,1:ncontext), wx(:,ncontext+1:end));

        fval = obj_fn(P);

        wgrad = wx(:,ncontext+1:end) * (P-Q);
        xgrad = wx(:,1:ncontext) * (P-Q)';
        grad = [wgrad, xgrad];

        grad = grad(:);
    end

    function P = P_fn(w, x)
        P = exp(x' * w);
        P = bsxfun(@rdivide, P, sum(P));
    end

    function res = obj_fn(P)
        v = zeros(ncontext,1);
        for j = 1:ncontext
            v(j) = kldiv(Q(:,j),P(:,j));
        end
        res = sum(v);
    end
   
    function res = kldiv(p,q)
        filt = p > 0;
        res = sum(p(filt) .* log(p(filt) ./ q(filt)));
    end
end
