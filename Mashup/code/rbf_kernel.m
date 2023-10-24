function K = rbf_kernel(X, Y, g)
  if ~exist('g','var')
    g = 1 / size(X, 2);
  end
  K = exp(-g * (bsxfun(@plus, sum(X.^2, 2), sum(Y.^2, 2)') - 2 * X * Y'));
end
