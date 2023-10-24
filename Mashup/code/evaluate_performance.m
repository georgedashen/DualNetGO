function [acc, f1, auprc] = evaluate_performace(class_score, label)
  alpha = 3;

  label = label > 0;

  [ncase nclass] = size(class_score);

  [~,o] = sort(class_score,2,'descend');
  p = sub2ind(size(label),(1:ncase)',o(:,1));
  acc = mean(label(p));

  a = repmat((1:ncase)',1,alpha);
  pred = sparse(a, o(:,1:alpha), 1, size(label,1),size(label,2));

  tab = crosstab([0;1;pred(:)],[0;1;label(:)]) - eye(2);
  f1 = 2*tab(2,2) / (2*tab(2,2)+tab(1,2)+tab(2,1));

  [~, auprc] = auc(label(:), class_score(:));
end
