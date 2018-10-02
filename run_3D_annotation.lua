
-- path to NetworkTraining
local codeDir="../" 
local dataDir="../MRAdata/"
local logdir="log_3D_annotation"
local nClasses=2
local bSize=3

package.path=package.path..";"..paths.concat(codeDir,"?.lua")..
                         ";"..paths.concat(paths.concat(codeDir,"?"),"init.lua")

require "nn"
require "cunn"
require "cudnn"
require "NetworkTraining"
local net=dofile("unet_v1_1ch.lua")
net:cuda()
local netParts={net}
local opts = {inplace=true, mode='training'}
optnet = require 'optnet'
local input=torch.CudaTensor(1,1,64,64,64)
optnet.optimizeMemory(net, input, opts)
input=nil

local imgDir=paths.concat(dataDir,"data/img")
local lblDir=paths.concat(dataDir,"data/lbl")

-- defines trainFiles and testFiles, tables of file names
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
  local i=i:float()
  -- flip
  if math.random()>0.5 then i:copy(image.flip(i,2)) l:copy(image.flip(l,1)) end
  if math.random()>0.5 then i:copy(image.flip(i,3)) l:copy(image.flip(l,2)) end
  if math.random()>0.5 then i:copy(image.flip(i,4)) l:copy(image.flip(l,3)) end
  -- randomly permute directions
  local perminds=torch.Tensor{1,2,3,4}
  local pi=torch.randperm(3)
  perminds:narrow(1,2,3):copy(torch.add(pi,1))
  local ip=i:permute(unpack(perminds:totable()))
  local lp=l:permute(unpack(pi:totable()))
  ic,lc= NetworkTraining.jitterImgLbl(ip,lp,torch.Tensor{64,64,64}) 
  
  lc:add(1)
  return ic,lc
end

local function testPreproc(i,l)
  return i,l:add(1)
end

local function setupDataProvider()
  local datasetTrain=NetworkTraining.DatasetDisk
    (imgDir,lblDir,trainFiles,augmentData)
  local dataProviderTrain=NetworkTraining.DataProviderDataset{
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
    noData=#trainFiles,
    bSize=bSize,
    prefix="basic"
  }
  local lF1=NetworkTraining.LoggerF1{
    logdir=logdir,
    transformOutputAndTarget=tot,
    prefix="f1Train"
  }
  return NetworkTraining.LoggerComposit{lBasic,lF1}
end

local function setupTester()
  local lTest=NetworkTraining.LoggerF1{
      logdir=logdir,
      transformOutputAndTarget=tot,
      prefix="test",
      saveBest=true
  }
  local datasetTest=NetworkTraining.DatasetDisk
    (imgDir,lblDir,testFiles,testPreproc) 
  local s=80
  local m=22
  -- the number of windows per file, here below, should be computed based on the image size and 
  -- the selected size of the window as 
  -- ceil[(sz(1)-2*m(1))/(s(1)-2*m(1))] * ceil[(sz(2)-2*m(2))/(s(2)-2*m(2))] * ceil[(sz(3)-2*m(3))/(s(3)-2*m(3))] 
  -- where
  --   sz is the size of the volumes as existing in the dataset, 
  --   s is the size of the crop forwarded through the network, s(1)=s(2)=s(3)=s
  --   m is the requested margin size, m(1)=m(2)=m(3)=m
  -- here it is hard-coded
  local dt2=NetworkTraining.DatasetPieces(datasetTest,3*11*8,torch.Tensor{s,s,s},torch.Tensor{m,m,m})
  local dataProviderTest=(NetworkTraining.DataProviderDataset{dataset=dt2})
  local testerTest=NetworkTraining.Tester{
    dataProvider=dataProviderTest:cuda(),
    logger=lTest,
    bSize=bSize
  }
  return testerTest
end

function calculateFrequency(fileList,dir)
  --print"calculating class frequency"
  local hist=torch.Tensor(nClasses+1):zero()
  local fn
  for k,f in pairs(fileList) do
    fn=paths.concat(dir,f)
    --print(fn)
    local lbl=torch.load(fn)
    hist:add(hist,1,lbl:double():histc(nClasses+1,0,nClasses))
  end
  --print("the histogram ",hist)
  return hist
end
  
local function setupCriterion()
  local classFreq=calculateFrequency(trainFiles,lblDir)
  -- index 1 corresponds to the ignored class - discard it
  classFreq=classFreq:div(classFreq:narrow(1,2,nClasses):sum())
  -- i found this method in the eNet implementation
  -- the weight is 1/log(f+1.02) where f is the corresp frequency
  classFreq:add(classFreq,1.02)
  local logClassFreq=torch.log(classFreq)
  logClassFreq:pow(logClassFreq,-1)
  local weights=torch.Tensor(nClasses+1)
  weights:copy(logClassFreq)
  weights[1]=0
  local crit=cudnn.VolumetricCrossEntropyCriterion(weights)
  return nn.ModuleCriterion(crit,nn.Padding(1,-2,4)):cuda()
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
print("the log dir is ",segmTask.logger.logdir)

-- run training
setup:Train(50000)
--[[
for i=1,3 do
  setup:Train(20000)
  o.learningRate=o.learningRate/2
end
--]]

