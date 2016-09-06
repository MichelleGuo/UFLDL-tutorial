function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
% hidden layer
for i = 1:numHidden
    if (i == 1)
        z = stack{1}.W*data;
    else
        z = stack{i}.W * data + hAct{i-1};
    end
    z = bsxfun(@plus,z,stack{i}.b);
    hAct{i}=sigmf(z,[1,0]);
end

% softmax layer feature
h_output = (stack{numHidden+1}.W * hAct{numHidden});
h_output = bsxfun(@plus,h_output,stack{numHidden+1}.b);
e_h = exp(h_output);
pred_prob = bsxfun(@rdivide,e_h,sum(e_h,1));
hAct{numHidden+1} = pred_prob;

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
% softmax layer cost
ceCost = 0;
c = log(pred_prob);
I=sub2ind(size(c),labels',1:size(c,2));
% find out matrix c' index, rows are depended on labels, columns are set by
% size(c,2)
values = c(I);
ceCost = -sum(values);

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
% cross entropy gradient
d = zeros(size(pred_prob));  
d(I) = 1;
error = pred_prob - d;

% gradient, error BP
% for i = nBegin:nStep:nEnd; nStep default = 1
% TODO: Review
for j = numHidden+1 : -1 : 1
    gradStack{j}.b = sum(error,2);
    if (j==1)
        gradStack{j}.W=error*data';
        break;
    else
        gradStack{j}.W = error*hAct{j-1}';
    end
    error = (stack{j}.W)'*error .*hAct{j-1}.* (1-hAct{j-1});
    % activation derivatives
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

wCost = 0;  
for l = 1:numHidden+1  
    wCost = wCost + .5 * ei.lambda * sum(stack{l}.W(:) .^ 2); 
end  
  
cost = ceCost + wCost;  
  
% Computing the gradient of the weight decay.  
for l = numHidden : -1 : 1  
    gradStack{l}.W = gradStack{l}.W + ei.lambda * stack{l}.W;
end  

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



