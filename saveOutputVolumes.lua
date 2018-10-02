-- path to NetworkTraining
local codeDir="../" 
local dataDir="../MRAdata/"
local imgDir=paths.concat(dataDir,"data/img")
local logdir="log_mip_annotation"
local netname="test_net_bestF1.t7" --"basic_net_last.t7"
local outdir=paths.concat(logdir,netname:sub(1,-4).."_output")
-- define testFiles
dofile(paths.concat(dataDir,"dataSplit.lua"))

package.path=package.path..";"..paths.concat(codeDir,"?.lua")..
                         ";"..paths.concat(paths.concat(codeDir,"?"),"init.lua")
require "cutorch" 
require "nn" 
require "cunn" 
require "cudnn"
require "image"
require "NetworkTraining"

net=torch.load(paths.concat(logdir,netname))
net:clearState();
os.execute("mkdir "..outdir)
net:evaluate()
--[[
-- for evaluation in the training mode
-- (BN with sample mean and variance as opposed to the running mean and var)
net:training()
dos=net:findModules('nn.VolumetricDropout') --dropouts
print(dos)
for k,v in pairs(dos) do v.p=0 end
net:training()
--]]

function testPreproc(input)
  return input
end

function savePngStack(img,dirname)
  os.execute("mkdir "..dirname)
  for k=1, img:size(1) do
    image.save(paths.concat(dirname,string.format("%03d.png",k)),img[{k,{},{}}])
  end
end

function process_output(o)
    local eo=o:clone():mul(-1):exp()
    local pr=torch.pow(torch.add(eo,1),-1)
    return pr:reshape(1,o:size(3),o:size(4),o:size(5))
end

for _,f in pairs(testFiles) do
    print("processing",f)
    local fn=paths.concat(imgDir,f)
    local input=torch.load(fn)
    input=input:reshape(1,input:size(1),input:size(2),input:size(3),input:size(4))
    local sz=input:size()
    output=torch.FloatTensor(sz)
    NetworkTraining.piecewiseForward(testPreproc(input),output,3,96,22,net,{[1]={},[2]={}})
    --torch.save(paths.concat(outdir,f),output)
    savePngStack(process_output(output):squeeze(),paths.concat(outdir,f))
end 
