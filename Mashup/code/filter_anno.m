function anno = filter_anno(anno, thres)
  anno = double(anno > 0);
  termsize = sum(anno, 2);
  
  [~, ord] = sort(termsize, 'ascend');
  anno = anno(ord,:);

  in = anno * anno';
  un = size(anno, 2) - (1 - anno) * (1 - anno)';
  jacc = in ./ un;

  max_jacc = max(triu(jacc, 1), [], 2);
  anno = anno(max_jacc <= thres,:);
end
