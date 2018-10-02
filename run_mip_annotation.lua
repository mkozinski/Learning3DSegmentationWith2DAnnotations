-- path to NetworkTraining
local codeDir="../" 
local dataDir="../MRAdata/"
local logdir="log_mip_annotation"
local nClasses=2
local bSize=3

package.path=package.path..";"..paths.concat(codeDir,"?.lua")..
                         ";"..paths.concat(paths.concat(codeDir,"?"),"init.lua")

require "nn"
require "cunn"
require "cudnn"
require "NetworkTraining"
require "criterionMIPhull"
require "jitterProjections"
require "DataProviderProjections"
local net=dofile("unet_v1_1ch.lua"):cuda()
local netParts={net}
local opts = {inplace=true, mode='training'}
optnet = require 'optnet'
local input=torch.CudaTensor(1,1,64,64,64)
optnet.optimizeMemory(net, input, opts)
input=nil

local trainImgDir=paths.concat(dataDir,"data/img_cut")
local testImgDir=paths.concat(dataDir,"data/img")
local trainLblDir=paths.concat(dataDir,"data/lbl_MIP_cut")
local testLblDir=paths.concat(dataDir,"data/lbl")

-- defines trainFiles, cutTrainFiles and testFiles, tables of file names
dofile(paths.concat(dataDir,"dataSplit.lua"))

-- transpose output and target for logging
-- map ground truth indexes and indexes of output vars
local function tot(o,t)
  p=o:permute(1,3,4,5,2):reshape(o:numel())
  q=p:mul(-1):exp()
  r=torch.add(q,1):pow(-1)
  u=t:reshape(t:numel()) 
      :add(-1)
  return r,u
end

local function augmentData(i,l) 
  -- pretend we have no annotation of the 3rd projection 
  l[3]:fill(0)
  local ic=i:float()
  local lc=l
  -- flip
  if math.random()>0.5 then ic:copy(image.flip(ic,2)) lc[2]:copy(image.flip(lc[2],1)) lc[3]:copy(image.flip(lc[3],1)) end
  if math.random()>0.5 then ic:copy(image.flip(ic,3)) lc[1]:copy(image.flip(lc[1],1)) lc[3]:copy(image.flip(lc[3],2)) end
  if math.random()>0.5 then ic:copy(image.flip(ic,4)) lc[1]:copy(image.flip(lc[1],2)) lc[2]:copy(image.flip(lc[2],2)) end
  -- randomly permute directions
  local perminds=torch.Tensor{1,2,3,4}
  local pi=torch.randperm(3)
  lp={}
  if pi[2]>pi[3] then lp[1]=lc[pi[1]]:transpose(1,2) else lp[1]=lc[pi[1]] end
  if pi[1]>pi[3] then lp[2]=lc[pi[2]]:transpose(1,2) else lp[2]=lc[pi[2]] end
  if pi[1]>pi[2] then lp[3]=lc[pi[3]]:transpose(1,2) else lp[3]=lc[pi[3]] end
  perminds:narrow(1,2,3):copy(torch.add(pi,1))
  local ip=ic:permute(unpack(perminds:totable()))
  -- jitter (random crop)
  ij,lj= jitterProjections(ip,lp,torch.Tensor{64,64,64},true) 

  for k=1,3 do
    lj[k]:add(1)
  end
  return ij,lj
end

local function testPreproc(i,l)
  return i,l:add(1)
end

local function setupDataProvider()
  local datasetTrain=NetworkTraining.DatasetDisk
    (trainImgDir,trainLblDir,cutTrainFiles,augmentData)
  local dataProviderTrain=DataProviderProjections{
    dataset=datasetTrain,
    ignoreLast=true,
    shuffle=true
  }
  return dataProviderTrain:cuda()
end

local function setupLogger()
  local lBasic=NetworkTraining.LoggerBasic{
    logdir=logdir,
    nParams=#netParts,
    saveEvery=100, --updates
    noData=#cutTrainFiles,
    bSize=bSize,
    prefix="basic"
  }
  return lBasic
end

local function setupTester()
  local lTest=NetworkTraining.LoggerF1{
      logdir=logdir,
      transformOutputAndTarget=tot,
      prefix="test",
      saveBest=true
  }
  local datasetTest=NetworkTraining.DatasetDisk
    (testImgDir,testLblDir,testFiles,testPreproc) 
  local s=80
  local m=22
  -- the number of windows per file, here below, should be computed based on the image size and 
  -- the selected size of the window as 
  -- ceil[(sz(1)-2*m(1))/(s(1)-2*m(1))] * ceil[(sz(2)-2*m(2))/(s(2)-2*m(2))] * ceil[(sz(3)-2*m(3))/(s(3)-2*m(3))] 
  -- where
  --   sz is the size of the volumes as existing in the dataset, 
  --   s is the size of the crop forwarded through the network, s(1)=s(2)=s(3)=s
  --   m is the requested margin size, m(1)=m(2)=m(3)=m
  -- here it is hard-coded as 3*11*8
  local dt2=NetworkTraining.DatasetPieces(datasetTest,3*11*8,torch.Tensor{s,s,s},torch.Tensor{m,m,m})
  local dataProviderTest=(NetworkTraining.DataProviderDataset{dataset=dt2})
  local testerTest=NetworkTraining.Tester{
    dataProvider=dataProviderTest:cuda(),
    logger=lTest,
    bSize=bSize
  }
  return testerTest
end

local function setupCriterion()
  return CriterionMIPhull():cuda() 
end

--create an instance of Task
segmTask=NetworkTraining.Task{
  net=net,
  netParts=netParts,
  n_fbs=1,-- number of batches processed before weights update
  test_every=50, --epochs
  optimEngine=optim.adam,
  bSize=bSize,
  optimState=
  {
    { learningRate = 1e-5,
      weightDecay = 1e-4,
      momentum = 0.9,
      learningRateDecay = 0.25e-5}
  },
  dataProvider=setupDataProvider(),
  logger=setupLogger(),
  tester=setupTester(),
  crit=setupCriterion()
}

-- create an instance of Setup
local tasks={segmTask}
setup=NetworkTraining.Setup(tasks)
collectgarbage()

-- save the current file to the log directory, to know all the parameters
local info = debug.getinfo(1,'S');
-- info.source contains the name of the current file
local cfn=info.source:sub(2,#info.source)
os.execute('cp '..cfn..' '..paths.concat(logdir,'setup.lua'))

-- for convenience
t=setup.tasks[1]
o=t.optimState[1]
print("the log dir is",segmTask.logger.logdir)

-- run training
setup:Train(50000)
--[[
for i=1,3 do
  setup:Train(20000)
  o.learningRate=o.learningRate/2
end
--]]

