--[[
  receptive field of this network is 44x44x44
--]]
require 'nn'
require 'cudnn'

local unet=nn.Sequential()

function getUpscaleConvolution(s)
  local c=cudnn.VolumetricConvolution(s,2*s, 1,1,1, 1,1,1, 0,0,0):noBias()
  c.weight:zero()
  for i=1,s do
    c.weight[2*i-1][i]=1
    c.weight[2*i][i]=1
  end
  return c
end
function getDownscaleConvolution(s)
  s=s/2
  local c=cudnn.VolumetricConvolution(2*s,s, 1,1,1, 1,1,1, 0,0,0):noBias()
  c.weight:zero()
  for i=1,s do
    c.weight[i][2*i-1]=0.5
    c.weight[i][2*i]=0.5
  end
  return c
end
-- building block
local function block(net,ni,no)
  net:add(cudnn.VolumetricConvolution(ni,no, 3,3,3, 1,1,1, 1,1,1))
  net:add(cudnn.VolumetricBatchNormalization(no))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.VolumetricConvolution(no,no, 3,3,3, 1,1,1, 1,1,1))
  net:add(cudnn.VolumetricBatchNormalization(no))
  net:add(cudnn.ReLU(true))
  return net
end
local function block2Up(net,ni)
  local no=2*ni
  local n=nn.Sequential()
  n:add(cudnn.VolumetricConvolution(ni,no, 3,3,3, 1,1,1, 1,1,1))
  n:add(cudnn.VolumetricBatchNormalization(no))
  n:add(cudnn.ReLU(true))
  n:add(cudnn.VolumetricConvolution(no,no, 3,3,3, 1,1,1, 1,1,1))
  n:add(cudnn.VolumetricBatchNormalization(no))
  n:add(cudnn.ReLU(true))
  n:add(nn.VolumetricDropout(0.1))
  net:add(nn.ConcatTable():add(getUpscaleConvolution(ni)):add(n))
  net:add(nn.CAddTable())
  return net
end
local function block2Down(net,ni)
  local no=ni/2
  local n=nn.Sequential()
  n:add(cudnn.VolumetricConvolution(ni,no, 3,3,3, 1,1,1, 1,1,1))
  n:add(cudnn.VolumetricBatchNormalization(no))
  n:add(cudnn.ReLU(true))
  n:add(cudnn.VolumetricConvolution(no,no, 3,3,3, 1,1,1, 1,1,1))
  n:add(cudnn.VolumetricBatchNormalization(no))
  n:add(cudnn.ReLU(true))
  n:add(nn.VolumetricDropout(0.1))
  net:add(nn.ConcatTable():add(getDownscaleConvolution(ni)):add(n))
  net:add(nn.CAddTable())
  return net
end

local function uBlock(net,n)
  local net0=nn.Sequential()
  net0:add(cudnn.VolumetricMaxPooling(2,2,2, 2,2,2))
  block2Up(net0,n)
  net0:add(nn.ConcatTable():add(nn:Identity()):add(net))
  net0:add(nn.JoinTable(2))
  block2Down(net0,4*n)
  net0:add(cudnn.VolumetricFullConvolution(2*n,n, 2,2,2, 2,2,2))
  return net0
end

local function innerBlock(ni,no)
  local net0=nn.Sequential()
  net0:add(cudnn.VolumetricMaxPooling(2,2,2, 2,2,2))
  block2Up(net0,ni)
  net0:add(cudnn.VolumetricFullConvolution(no,ni, 2,2,2, 2,2,2))
  return net0
end

block(unet,1,64)
unet:add(nn.ConcatTable():add(nn.Identity()):add(uBlock(innerBlock(128,256),64)))
unet:add(nn.JoinTable(2))
block2Down(unet,128)
unet:add(cudnn.VolumetricConvolution(64,1, 1,1,1))

return unet

