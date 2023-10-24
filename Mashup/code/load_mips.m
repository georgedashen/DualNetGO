%$ onttype: level1, level2, level3
%% genes: cell array of query gene names
%%
function anno = load_mips(onttype, genes)
  mips_path = 'data/annotations/yeast';
  mips_genes = textread(sprintf('%s/yeast_mips_%s_genes.txt', mips_path, onttype), '%s');
  filt = ismember(genes, mips_genes);
  
  mips_terms = textread(sprintf('%s/yeast_mips_%s_terms.txt', mips_path, onttype), '%s');
  [g t] = textread(sprintf('%s/yeast_mips_%s_adjacency.txt', mips_path, onttype), '%d%d');
  mips_anno = sparse(t, g, true, length(mips_terms), length(mips_genes)) > 0;
  
  anno = zeros(length(mips_terms), length(genes));
  genemap = containers.Map(mips_genes, 1:length(mips_genes));
  s2goind = cell2mat(values(genemap, genes(filt)));
  anno(:,filt) = mips_anno(:,s2goind);
end
