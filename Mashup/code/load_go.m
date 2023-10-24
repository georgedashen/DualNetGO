%% gotype: 'bp', 'mf', or 'cc'
%% genes: cell array of query gene names
%% ontsize (optional): [min_size max_size], only return terms within a size range
%% no_overlap (optional): remove redundant terms
%%
function anno = load_go(gotype, genes, ontsize, no_overlap)
  prefix = 'go_human_ref';
  
  go_path = 'data/annotations/human';
  gogene = textread(sprintf('%s/%s_genes.txt', go_path, prefix), '%s');
  filt = ismember(genes, gogene);
  
  goterm = textread(sprintf('%s/%s_%s_terms.txt', go_path, prefix, gotype), '%s');
  [g t] = textread(sprintf('%s/%s_%s_adjacency.txt', go_path, prefix, gotype), '%d%d');
  goanno = sparse(t, g, true, length(goterm), length(gogene));
  
  anno = zeros(length(goterm), length(genes));
  genemap = containers.Map(gogene, 1:length(gogene));
  s2goind = cell2mat(values(genemap, genes(filt)));
  anno(:,filt) = goanno(:,s2goind);
  
  %% Use ontology graph to propagation annotations
  termfile = sprintf('%s/graph/go_%s.terms', go_path, gotype);
  nterm = length(textread(termfile, '%s'));
  
  mapfile = sprintf('%s/graph/go_%s.map', go_path, gotype);
  [t, i] = textread(mapfile, '%s\t%d');
  m = containers.Map(t, i);
  
  linkfile = sprintf('%s/graph/go_%s.links', go_path, gotype);
  M = dlmread(linkfile);
  ontgraph = sparse(M(:,1), M(:,2), true, nterm, nterm);
  reachable = get_reachable(ontgraph);
  
  goind = cell2mat(values(m, goterm));
  anno = (reachable(:,goind) * anno) > 0; % propagate
  anno = anno(sum(anno, 2) > 0,:);

  %% Select terms in the given size range
  if exist('ontsize', 'var')
    term_size = sum(anno, 2);
    filt = ontsize(1) <= term_size & term_size <= ontsize(2);
    anno = anno(filt,:);
  end

  %% Remove redundant terms
  if exist('no_overlap', 'var') && no_overlap
    anno = filter_anno(anno, 0.1);
  end
end
