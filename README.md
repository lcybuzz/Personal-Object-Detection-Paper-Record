# Personal-Object-Detection-Paper-Record
# Under construction!
# Table of Contents
- [Deep Learning Methods](#deep-learning-methods)
  - [Popular methods](#popular-methods)
  - [One Stage](#one-stage)
  - [Other Interesting Methods](#other-interesting-methods)
- [Taditional Classical Methods](#traditional-classical-methods)
- [Datasets](#datasets)
- [Leaderboards](#leaderboards)
- [Sources-Lists](#sources-lists)

# Deep Learning Methods
## Popular methods

### **SPPNet**
**[Paper]** Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition <Br>
**[Year]** ECCV 2014 / TPAMI 2015 <Br>
**[Authors]** 	[Kaiming He](http://kaiminghe.com/), [Xiangyu Zhang](https://www.cs.purdue.edu/homes/xyzhang/),[Shaoqing Ren](http://shaoqingren.com/), [Jian Sun](http://www.jiansun.org/)  <Br>
**[Pages]** https://github.com/ShaoqingRen/SPP_net  <Br>
**[Description]** <Br>

### **R-CNN**
**[Paper]** Rich feature hierarchies for accurate object detection and semantic segmentation <Br>
**[Year]** CVPR 2014 <Br>
**[Authors]** 		[Ross Girshick](http://www.rossgirshick.info/), [Jeff Donahue](http://jeffdonahue.com/),	[Trevor Darrell](http://people.eecs.berkeley.edu/~trevor/), [Jitendra Malik](http://people.eecs.berkeley.edu/~malik/)   <Br>
**[Pages]** https://github.com/rbgirshick/rcnn  <Br>
**[Description]** <Br>

### **Fast R-CNN**
**[Paper]** Fast R-CNN: Fast Region-based Convolutional Networks for object detection  <Br>
**[Year]** ICCV 2015 <Br>
**[Authors]** 		[Ross Girshick](http://www.rossgirshick.info/) <Br>
**[Pages]** https://github.com/rbgirshick/fast-rcnn  <Br>
**[Description]** <Br>


### **Faster R-CNN**
**[Paper]** Fast R-CNN: Fast Region-based Convolutional Networks for object detection  <Br>
**[Year]** ICCV 2015 <Br>
**[Authors]**   	[Shaoqing Ren](http://shaoqingren.com/), [Kaiming He](http://kaiminghe.com/), [Ross Girshick](http://www.rossgirshick.info/), 	[Jian Sun](http://www.jiansun.org/)  <Br>
**[Pages]** <Br>
	https://github.com/rbgirshick/py-faster-rcnn  <Br>
	https://github.com/ShaoqingRen/faster_rcnn  <Br>
**[Description]** <Br>

## One Stage
### **YOLO ★★★**
**[Paper]** Fast R-CNN: Fast Region-based Convolutional Networks for object detection  <Br>
**[Year]** ICCV 2016 <Br>
**[Authors]**   	[Joseph Redmon](https://pjreddie.com/), [Santosh Divvala](http://allenai.org/team/santoshd/), [Ross Girshick](http://www.rossgirshick.info/), 	[Ali Farhadi](https://homes.cs.washington.edu/~ali/)  <Br>
**[Pages]** <Br>
	https://pjreddie.com/darknet/yolo/  <Br>
	https://github.com/pjreddie/darknet  <Br>
	https://github.com/gliese581gg/YOLO_tensorflow <Br>
	https://github.com/thtrieu/darkflow <Br>
**[Description]** <Br>
1) 一种one stage的目标检测方法, 用一个网络同时得出bounding box及其类别, 端到端, 速度快;
2) 将图像分为S*S个gird, 其作用类似于region proposal. 首先, 对于每个grid计算B个bounding box, 得到x, y, h, w以及confidence score, 此处confidence score是由bbox与真值的IoU和该grid存在目标的可能性之积组成的. 其次, 计算grid属于每个类别的概率, 通过bbox置信度类别概率相乘可得到bbox属于某一类的可能性;
3) loss设计: 位置loss(只考虑负责该目标的bbox), 置信度loss(负责目标的bbox和不包含目标的bbox的加权和), 分类 loss(只考虑出现了目标的grid), 所有loss均使用均方误差. 另外, 对位置loss给予更高的权重以平衡位置loss和类别loss的占比, 对h和w开根号降低大bbox对位置偏差的敏感度.

### ** SSD **
**[Paper]** SSD: Single Shot MultiBox Detector  <Br>
**[Year]** ECCV 2016 Oral <Br>
**[Authors]**   	[Wei Liu](http://www.cs.unc.edu/~wliu/), [Dragomir Anguelov](), [Christian Szegedy](https://research.google.com/pubs/ChristianSzegedy.html), [Scott Reed](http://www.scottreed.info/), [Cheng-Yang Fu](http://www.cs.unc.edu/~cyfu/), [Alexander C. Berg](http://acberg.com/). <Br>
**[Pages]** <Br>
	https://github.com/weiliu89/caffe/tree/ssd  <Br>
	https://github.com/balancap/SSD-Tensorflow  <Br>
**[Description]** <Br>

## Other Interesting Methods  


# Taditional Classical Methods


# Datasets

# Leaderboards
 PASCAL VOC http://host.robots.ox.ac.uk:8080/leaderboard/main_bootstrap.php <Br>
  ILSVRC2016 http://image-net.org/challenges/LSVRC/2016/results <Br>
# Sources-Lists
https://handong1587.github.io/deep_learning/2015/10/09/object-detection.html <Br>

