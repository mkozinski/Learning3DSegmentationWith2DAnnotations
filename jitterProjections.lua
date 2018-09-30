function jitterProjections(img,lbl,d,expand)
  --[[
    returns a random crop of img and lbl, of size d
    img is a 3d volume, sized no_channel X height X width X depth
    lbl is a table of three 2D images, 
    where lbl[i] is the annotation of img:max(i+1)
    d is the requested crop size
    expand is a boolean - if true, crops that exceed the spatial limits of img and lbl are allowed
  --]]
  local sz=torch.LongTensor(img:size())
  sz=sz:double()
  local marg=sz:narrow(1,2,sz:size(1)-1)-d
  if not expand then
    assert(torch.all(marg:ge(0)),"requested oversize crop. ")
  end
  local newsz=sz:clone()
  newsz:narrow(1,2,newsz:numel()-1):copy(d)
  assert(newsz:numel()-1==#lbl, "number of projections has to be equal input dimensionality +1")
  local newimg=img.new(newsz:long():storage()):zero()
  -- create the new label table
  local newlbl={}
  -- populate it with tensors of size of cropped projections
  for il=1,#lbl do
    local nsz=torch.LongTensor(lbl[il]:size())
    local lind=1
    for dimind=1,#lbl do
      if dimind~=il then -- the index to the label table indicates projection dimension
        nsz[lind]=newsz[dimind+1] -- +1 stands for the feature dimension
        lind=lind+1
      end
    end
    newlbl[il]=torch.Tensor(nsz:long():storage()):typeAs(lbl[il]):fill(0)
  end
  -- generate the random offsets
  local marg2=marg:clone()
  marg2:maskedFill(marg:lt(0),0)
  local off=torch.rand(d:size()):cmul(marg2):floor()
  -- create indexes for cropping 
  local srcind={}
  local dstind={}
  for i=1,d:numel() do
    local sind=math.max(1,math.floor(-marg[i]/2))
    local szi=math.min(sz[i+1],d[i])
    table.insert(dstind,{sind,sind+szi-1})
    table.insert(srcind,{off[i]+1,off[i]+szi})
  end
  -- crop img and lbl
  for il=1,#newlbl do
   local di=table.remove(dstind,il) 
   local si=table.remove(srcind,il) 
   newlbl[il][dstind]:copy(lbl[il][srcind])
   table.insert(dstind,il,di)
   table.insert(srcind,il,si)
  end
  table.insert(dstind,1,{})
  table.insert(srcind,1,{})
  newimg[dstind]:copy(img[srcind]:clone())
  return newimg,newlbl
end
