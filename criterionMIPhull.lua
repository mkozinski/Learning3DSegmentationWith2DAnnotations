require 'nn'
require 'cudnn'

local criterion, parent = torch.class('CriterionMIPhull', 'nn.Criterion')
--[[
  this is a loss function for 3D predictions and annotations of 2D projections
  the concept is simple: the prediction is projected along the 3 principl axes
  and the projections are evaluated against projection annotations

  the complicating factor is the visual-hull-based filtering
  it is intended to remove annotations that are not consistent across different views
  which may result when the training data (and 2D annotations) are cropped
  (cropped annotations of projections contain labels of foreground structures
  that are situated outside of the 3D crop)
  this is however not very essential from a conceptual view point

  supports (up to) three annotated projections
  assumes batch input
  assumes second dimension of target indexes projections on thre principal dimensions
  assumes class indeces are 1 - "ignored" area around centerlines
                            2 - background
                            3 - foreground
]]--

function criterion:__init(weight)
  parent.__init(self)
  -- cross entropies for each of the three projections
  self.criterion_1=cudnn.SpatialCrossEntropyCriterion(weight)
  self.criterion_2=cudnn.SpatialCrossEntropyCriterion(weight)
  self.criterion_3=cudnn.SpatialCrossEntropyCriterion(weight)
 
  local n=nn.ConcatTable()
  -- max to project the prediction, 
  -- padding to transit from one to three output channels
  -- (ignored, background, foreground)
  -- (this is a technicality following from torch implementation of crossentropy)
  n:add(nn.Sequential():add(nn.Max(3)):add(nn.Padding(1,-2,3)))
  n:add(nn.Sequential():add(nn.Max(4)):add(nn.Padding(1,-2,3)))
  n:add(nn.Sequential():add(nn.Max(5)):add(nn.Padding(1,-2,3)))
  -- used to project the prediction
  self.net=n
  -- it is easier to do self.proj:forward(t) than t:max()
  -- because the former removes the maxed-out dimension
  self.proj1=nn.Max(1,3)
  self.proj2=nn.Max(2,3)
  self.proj3=nn.Max(3,3)
  -- p1, p2 p3 are the projected annotations
  self.p1=torch.Tensor()
  self.p2=torch.Tensor()
  self.p3=torch.Tensor()
end

function criterion:visualHullFilter(t1,t2,t3)
  --[[
    takes the MIP annotations corresponding to the projections along axes 1, 2, 3
    filters away some false positive annotations in these projections
    by building visual hulls of the annotations and re-projecting them back to the images
    a separate hull is build for the foreground annotation (index 3)
    and for a "ignore"-class annotation (index 1)
    projections of these two hulls are then combined
  --]]
  local h=t1.new(t1:size(1), --batch dimension
                 t2:size(2), -- the first spatial dimension
                 t3:size(3),
                 t1:size(3)):fill(1)
  local hullIg=h:clone() -- a visual hull for the ignored region around annotation centerlines
  local hullFg=h -- a visual hull for the foreground annotation
  hullIg:maskedFill(t1:eq(2):reshape(t1:size(1),1,t1:size(2),t1:size(3)):expandAs(hullIg),0)
  hullIg:maskedFill(t2:eq(2):reshape(t2:size(1),t2:size(2),1,t2:size(3)):expandAs(hullIg),0)
  hullIg:maskedFill(t3:eq(2):reshape(t3:size(1),t3:size(2),t3:size(3),1):expandAs(hullIg),0)
  hullFg:maskedFill(t1:lt(3):reshape(t1:size(1),1,t1:size(2),t1:size(3)):expandAs(hullFg),0)
  hullFg:maskedFill(t2:lt(3):reshape(t2:size(1),t2:size(2),1,t2:size(3)):expandAs(hullFg),0)
  hullFg:maskedFill(t3:lt(3):reshape(t3:size(1),t3:size(2),t3:size(3),1):expandAs(hullFg),0)
  
  self.proj1:forward(hullIg)
  self.p1:resizeAs(self.proj1.output)
  self.p1:mul(self.proj1.output,-1) -- p1 contains 0 for bg, -1 for ignored and foreground pixels
  self.p1:add(self.p1,2,self.proj1:forward(hullFg)):add(2) -- 1 for ignored, 2 for background, 3 for foreground

  self.proj2:forward(hullIg)
  self.p2:resizeAs(self.proj2.output)
  self.p2:mul(self.proj2.output,-1)
  self.p2:add(self.p2,2,self.proj2:forward(hullFg)):add(2)

  self.proj3:forward(hullIg)
  self.p3:resizeAs(self.proj3.output)
  self.p3:mul(self.proj3.output,-1)
  self.p3:add(self.p3,2,self.proj3:forward(hullFg)):add(2)

  return self.p1,self.p2,self.p3
end

function criterion:updateOutput(input, target)
  assert(input:dim() == 5, 'mini-batch supported only')
  assert(target[1]:dim() == 3, 'mini-batch supported only')
  assert(target[2]:dim() == 3, 'mini-batch supported only')
  assert(target[3]:dim() == 3, 'mini-batch supported only')
  assert(input:size(1) == target[1]:size(1), 'input and target should have same batch size')
  assert(input:size(2) == 1, 'single channel input expected')
  assert(input:size(3) == target[3]:size(2), 'input and target should be of same size')
  assert(input:size(4) == target[1]:size(2), 'input and target should be of same size')
  assert(input:size(5) == target[2]:size(3), 'input and target should be of same size')
  self.t1,self.t2,self.t3=self:visualHullFilter(target[1],target[2],target[3])
  local mp=self.net:forward(input)
  self.output=self.criterion_1:forward(mp[1],self.t1)
             +self.criterion_2:forward(mp[2],self.t2)
             +self.criterion_3:forward(mp[3],self.t3)

  return self.output
end

function criterion:updateGradInput(input, target)
  --[[
  assumes last input and target are still relevant 
  ]]--
  assert(input:dim() == 5, 'mini-batch supported only')
  assert(target[1]:dim() == 3, 'mini-batch supported only')
  assert(target[2]:dim() == 3, 'mini-batch supported only')
  assert(target[3]:dim() == 3, 'mini-batch supported only')
  assert(input:size(1) == target[1]:size(1), 'input and target should have same batch size')
  assert(input:size(2) == 1, 'single channel input expected')
  assert(input:size(3) == target[3]:size(2), 'input and target should be of same size')
  assert(input:size(4) == target[1]:size(2), 'input and target should be of same size')
  assert(input:size(5) == target[2]:size(3), 'input and target should be of same size')
  self.criterion_1:backward(self.net.output[1],self.t1)
  self.criterion_2:backward(self.net.output[2],self.t2)
  self.criterion_3:backward(self.net.output[3],self.t3)
  self.net:backward(input,{self.criterion_1.gradInput,self.criterion_2.gradInput,self.criterion_3.gradInput})
  self.gradInput = self.net.gradInput

  return self.gradInput
end

function criterion:type(type)
  if type then
    self.criterion_1:type(type)
    self.criterion_2:type(type)
    self.criterion_3:type(type)
    self.net:type(type)
    self.proj1:type(type)
    self.proj2:type(type)
    self.proj3:type(type)
    self.p1:type(type)
    self.p2:type(type)
    self.p3:type(type)
  end
  parent.type(self, type)
  return self
end
