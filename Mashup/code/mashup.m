%% network_files (cell array): paths to adjacency list files
%% ngene (int): number of genes in input networks
%% ndim (int): number of output dimensions 
%% svd_approx (bool): whether to use SVD approximation for large-scale networks
%%
function x = mashup(network_files, ngene, ndim, svd_approx)
  if svd_approx
    RR_sum = zeros(ngene);
    for i = 1:length(network_files)
      fprintf('Loading %s\n', network_files{i});
      A = load_network(network_files{i}, ngene);
      fprintf('Running diffusion\n');
      Q = rwr(A, 0.5);
    
      R = log(Q + 1/ngene); % smoothing
      RR_sum = RR_sum + R * R';
    end
    clear R Q A
    
    fprintf('All networks loaded. Learning vectors via SVD...\n');
    [V, d] = eigs(RR_sum, ndim);
    x = diag(sqrt(sqrt(diag(d)))) * V';
  else
    Q_concat = [];
    for i = 1:length(network_files)
      fprintf('Loading %s\n', network_files{i});
      A = load_network(network_files{i}, ngene);
      fprintf('Running diffusion\n');
      Q = rwr(A, 0.5);

      Q_concat = [Q_concat; Q];
    end
    clear Q A
    Q_concat = Q_concat / length(network_files);

    fprintf('All networks loaded. Learning vectors via iterative optimization...\n');
    x = vector_embedding(Q_concat, ndim, 1000);
  end

  fprintf('Mashup features obtained.\n');

  function A = load_network(filename, ngene)
    M = dlmread(filename);
    A = full(sparse(M(:,1), M(:,2), M(:,3), ngene, ngene));
    if ~isequal(A, A') % symmetrize
      A = A + A';
    end
    A = A + diag(sum(A, 2) == 0);
  end
end
