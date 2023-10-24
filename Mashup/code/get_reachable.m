function R = get_reachable(A)
  nterm = size(A, 1);
  R = sparse(nterm, nterm);
  visit = speye(nterm);
  while nnz(visit) > 0
    R = R | visit;
    visit = double(visit) * A > 0;
  end
end
