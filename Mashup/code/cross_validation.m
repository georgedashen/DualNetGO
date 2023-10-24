function [acc, f1, aupr] = cross_validation(x, anno, nperm)
  % Parameters
  cvrep = 5;
  gvec = -3:1:0;
  cvec = -2:1:2;
  test_frac = 0.2;

  % Scale features
  maxval = max(x, [], 2);
  minval = min(x, [], 2);
  x = bsxfun(@times, bsxfun(@minus, x, minval), 1 ./ (maxval - minval));

  % Filter genes with no annotations
  filt = sum(anno) > 0;
  anno = anno(:,filt);
  x = x(:,filt);

  [nclass, ngene] = size(anno);

  acc = zeros(nperm, 1);
  f1 = zeros(nperm, 1);
  aupr = zeros(nperm, 1);
  for p = 1:nperm
    fprintf('[Trial #%d]\n', p);

    [ntest, test_filt] = cv_partition(anno, test_frac);
    ntrain = ngene - ntest;

    fprintf('Pregenerating kernels:\n');
    rbfK = cell(length(gvec), 1);
    for i = 1:length(gvec)
      fprintf('%d / %d ... ', i, length(gvec)); tic
      rbfK{i} = rbf_kernel(x', x', 10^gvec(i));
      fprintf('done. '); toc
    end

    % Set up nested CV data
    train_ind = find(~test_filt);
    ntest_nested = floor(ntrain * test_frac);
    test_ind_nested = cell(cvrep, 1);
    train_ind_nested = cell(cvrep, 1);
    for it = 1:cvrep 
      rp = randperm(ntrain, ntest_nested);
      test_ind_nested{it} = train_ind(rp);
      train_ind_nested{it} = train_ind;
      train_ind_nested{it}(rp) = [];
    end
    
    retmax = -inf;
    gmax = 1;
    cmax = 1;
    fprintf('Running nested cross validation...\n');
    for gi = 1:length(gvec)
      for ci = 1:length(cvec)
        tt = tic;

        cv_result = zeros(cvrep, 1);
        for it = 1:cvrep
          Ktrain = rbfK{gi}(train_ind_nested{it},train_ind_nested{it});
          Ktest = rbfK{gi}(test_ind_nested{it},train_ind_nested{it});

          class_score = zeros(ntest_nested, nclass);
          %parfor s = 1:nclass
          for s = 1:nclass
            Ytrain = full(double(anno(s,train_ind_nested{it})') * 2 - 1);
            Ytest = full(double(anno(s,test_ind_nested{it})') * 2 - 1);

            model = svmtrain(Ytrain, [(1:size(Ktrain,1))', Ktrain], ['-t 4 -b 1 -q -c ', num2str(10^cvec(ci))]);
            posind = find(model.Label > 0);
            if ~isempty(posind)
              [~, ~, dec] = svmpredict(Ytest, [(1:size(Ktest,1))', Ktest], model, '-q');
              class_score(:,s) = dec(:,posind);
            end
          end

          [~, ~, cv_result(it)] = evaluate_performance(class_score, anno(:,test_ind_nested{it})');
        end
        ret = median(cv_result);
        if retmax < ret
          retmax = ret;
          gmax = gi;
          cmax = ci;
        end

        fprintf('gi:%d, ci:%d, ret:%f, ', gi, ci, ret); toc(tt)
      end
    end

    fprintf('Using full training data...\n')
    Ktrain = rbfK{gmax}(~test_filt,~test_filt);
    Ktest = rbfK{gmax}(test_filt,~test_filt);

    class_score = zeros(ntest, nclass);
    %parfor s = 1:nclass
    for s = 1:nclass
      Ytrain = full(double(anno(s,~test_filt)') * 2 - 1);
      Ytest = full(double(anno(s,test_filt)') * 2 - 1);

      model = svmtrain(Ytrain, [(1:ntrain)', Ktrain], ['-t 4 -b 1 -q -c ', num2str(10^cvec(cmax))]);
      posind = find(model.Label > 0);
      if ~isempty(posind)
        [~, ~, dec] = svmpredict(Ytest, [(1:ntest)', Ktest], model, '-q');
        class_score(:,s) = dec(:,posind);
      end
    end

    test_data.class_score{p} = class_score;
    test_data.label{p} = anno(:,test_filt)';
    [acc(p), f1(p), aupr(p)] = evaluate_performance(class_score, anno(:,test_filt)');
    fprintf('[Trial #%d] acc: %f, f1: %f, aupr: %f\n', p, acc(p), f1(p), aupr(p));
  end

  function [ntest, test_filt] = cv_partition(anno, test_frac)
    ng = size(anno, 2);
    ntest = floor(ng * test_frac);
    test_ind = randperm(ng, ntest);
    test_filt = false(ng, 1);
    test_filt(test_ind) = true;
    ntest = length(test_ind);
  end
end
