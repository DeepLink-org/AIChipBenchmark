#!/bin/bash
cp -R samples samples-bak
dir=`dirname $0`/samples
echo $dir
sed -i "s/.cuda()//g" `grep -rl ".cuda()"  $dir`
sed -i "s/device=\"cuda\",/device='cpu',/g" `grep -rl "device=\"cuda\","  $dir`
sed -i "s/device='cuda',/device='cpu',/g" `grep -rl "device='cuda',"  $dir`

sed -i "s/, device=\"cuda\"/, device='cpu'/g" `grep -rl ", device=\"cuda\""  $dir`
sed -i "s/, device='cuda'/, device='cpu'/g" `grep -rl ", device='cuda'"  $dir`
sed -i "s/,device='cuda'/, device='cpu'/g" `grep -rl ",device='cuda'"  $dir`