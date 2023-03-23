#!/bin/bash
cp -R samples samples-bak
dir=`dirname $0`/samples
echo $dir
gsed -i "s/.cuda()//g" `grep -rl ".cuda()"  $dir`
gsed -i "s/device=\"cuda\",/device='cpu',/g" `grep -rl "device=\"cuda\","  $dir`
gsed -i "s/device='cuda',/device='cpu',/g" `grep -rl "device='cuda',"  $dir`

gsed -i "s/, device=\"cuda\"/, device='cpu'/g" `grep -rl ", device=\"cuda\""  $dir`
gsed -i "s/, device='cuda'/, device='cpu'/g" `grep -rl ", device='cuda'"  $dir`
gsed -i "s/,device='cuda'/, device='cpu'/g" `grep -rl ",device='cuda'"  $dir`