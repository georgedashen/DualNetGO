addpath code

%% Example parameters
org = 'human';      % use human or yeast data
onttype = 'level1'; % which type of annotations to use
                    %   options: {bp, mf, cc} for human GO,
                    %            {level1, level2, level3} for yeast MIPS
ontsize = [];       % consider terms in a specific size range (*human GO only*)
                    %   examples: [11 30], [31 100], [101 300]
nperm = 5;          % number of cross-validation trials
svd_approx = true;  % use SVD approximation for Mashup
                    %   recommended: true for human, false for yeast
ndim = 800;         % number of dimensions
                    %   recommended: 800 for human, 500 for yeast
ngene = 19385      ;      % 19385 for human, 21317 for mouse

%% Construct network file paths
string_nets = {'neighborhood', 'fusion', 'cooccurence', 'coexpression', ...
               'experimental', 'database', 'textmining'};
network_files = cell(1, length(string_nets));
for i = 1:length(string_nets)
  network_files{i} = sprintf('data/networks/%s_%s_edgeList.txt', ...
                             org, string_nets{i});
end


%% Mashup integration
fprintf('[Mashup]\n');
tic
x = mashup(network_files, ngene, ndim, svd_approx);
toc
save 'human_Mashup.mat' x


