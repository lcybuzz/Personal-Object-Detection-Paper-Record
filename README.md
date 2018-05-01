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

### **SPPNet ★★**
**[Paper]** Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition <Br>
**[Year]** ECCV 2014 / TPAMI 2015 <Br>
**[Authors]** 	[Kaiming He](http://kaiminghe.com/), [Xiangyu Zhang](https://www.cs.purdue.edu/homes/xyzhang/),[Shaoqing Ren](http://shaoqingren.com/), [Jian Sun](http://www.jiansun.org/)  <Br>
**[Pages]** https://github.com/ShaoqingRen/SPP_net  <Br>
**[Description]** <Br>
1) 提出了空间金字塔池化, 可以把任意尺寸的特征图pool成n*n大小. 设置不同的n, 即可得到不同尺度的feature, 形成金字塔. <Br>
2) SPP使网络支持任意大小的输入, 另外多尺度的特征使网络的性能有了显著提升. 后面的ROI pooling应该也是受了这个的启发. <Br>

### **R-CNN ★★★**
**[Paper]** Rich feature hierarchies for accurate object detection and semantic segmentation <Br>
**[Year]** CVPR 2014 <Br>
**[Authors]** 		[Ross Girshick](http://www.rossgirshick.info/), [Jeff Donahue](http://jeffdonahue.com/),	[Trevor Darrell](http://people.eecs.berkeley.edu/~trevor/), [Jitendra Malik](http://people.eecs.berkeley.edu/~malik/)   <Br>
**[Pages]** https://github.com/rbgirshick/rcnn  <Br>
**[Description]** <Br>
1) DL用于目标检测的首创. 用Selective Search产生约2k个proposal, 送入CNN得到类别和bounding box的位置调整. <Br>
2) RCNN需对每个proposal进行CNN特征提取, 速度慢. 

### **Fast R-CNN ★★**
**[Paper]** Fast R-CNN: Fast Region-based Convolutional Networks for object detection  <Br>
**[Year]** ICCV 2015 <Br>
**[Authors]** 		[Ross Girshick](http://www.rossgirshick.info/) <Br>
**[Pages]** https://github.com/rbgirshick/fast-rcnn  <Br>
**[Description]** <Br>
1) 提出ROI pooling. 对Selective Search后的proposal直接映射到feature map对应位置, 并用ROI pooling将每个proposal对应的特征区域resize到固定大小, 送入后面的全连接层得到类别和位置调整. 此处FC层使用了SVD分解提速. <Br>


### **Faster R-CNN ★★★**
**[Paper]** Fast R-CNN: Fast Region-based Convolutional Networks for object detection  <Br>
**[Year]** ICCV 2015 <Br>
**[Authors]**   	[Shaoqing Ren](http://shaoqingren.com/), [Kaiming He](http://kaiminghe.com/), [Ross Girshick](http://www.rossgirshick.info/), 	[Jian Sun](http://www.jiansun.org/)  <Br>
**[Pages]** <Br>
	https://github.com/rbgirshick/py-faster-rcnn  <Br>
	https://github.com/ShaoqingRen/faster_rcnn  <Br>
**[Description]** <Br>
1) 终于将提取Proposal, 特征提取和分类位置精修三个部分融合到一起. 提出了Region Proposal Network用于提取Proposal. <Br>
2) RPN事先定义了9个Anchor box， 用于处理不同尺寸和横纵比的目标. Loss包括分类(判断该anchor box是否包含目标)和回归(位置调整)两部分. <Br>
3) RPN得到Proposal后, 与Fast R-CNN步骤类似, 也通过ROI pooling将特征resize到固定大小并完成后续的分类及回归操作. <Br>
4) 训练时采用了交替训练RPN和Fast R-CNN的策略. 代码有时间应该仔细阅读. <Br>
**[Reference]** <Br>	
https://www.cnblogs.com/zf-blog/p/7273182.html <Br>

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

### **YOLO9000 ★★**
**[Paper]** YOLO9000: Better,Faster,Stronger  <Br>
**[Year]** CVPR 2017<Br>
**[Authors]**   [Joseph Redmon](https://pjreddie.com/), [Ali Farhadi](https://homes.cs.washington.edu/~ali/)  <Br>
**[Pages]** <Br>
	http://pjreddie.com/yolo9000/  <Br>
	https://github.com/philipperemy/yolo-9000  <Br>
	https://github.com/allanzelener/YAD2K <Br>
	https://github.com/hizhangp/yolo_tensorflow <Br>
	https://github.com/longcw/yolo2-pytorch <Br>
**[Description]** <Br>
1) YOLO的改进版本, 参考了SSD及Faster RCNN, 达到了更快更强的检测效果;
2) 卷积代替全连接层. 考虑到物体尤其是大目标倾向于出现在图像中间, 因此将网络输入从448改到416, 使得最后生成13*3的feature map, 这样就有一正中央的grid去预测物体位置;
3) 2个Bounding box改为Faster RCNN中的Anchor box. 然而此处的bbox不是预定义的, 而是用K-Means聚类出来的五个不同大小比例的bbox, 此处聚类距离用的是1-IoU以消除bbox大小对结果产生的影响;
4) 多尺度. 与SSD在多层的feature map上预测不同, 本文的做法是将26*26的feature map空间相邻元素concat起来, 即从26*26*512 -> 13*13*2048, 然后和下一层的map合并到一起. 另外, 在训练时不固定输入的尺寸, 而每隔10个batch改变输入的大小

### **YOLOv3 ★☆**
**[Paper]** YOLO9000: Better,Faster,Stronger  <Br>
**[Year]** arXiv 1804 <Br>
**[Authors]**   [Joseph Redmon](https://pjreddie.com/), [Ali Farhadi](https://homes.cs.washington.edu/~ali/)  <Br>
**[Pages]** <Br>
	https://pjreddie.com/darknet/yolo/ <Br>
**[Description]** <Br>
1) 粗读, 对前两个版本做了一些细节上的改进. 主要是多尺度预测和更好的backbone <Br>

### **SSD ★★★**
**[Paper]** SSD: Single Shot MultiBox Detector  <Br>
**[Year]** ECCV 2016 Oral <Br>
**[Authors]**   	[Wei Liu](http://www.cs.unc.edu/~wliu/), Dragomir Anguelov, [Christian Szegedy](https://research.google.com/pubs/ChristianSzegedy.html), [Scott Reed](http://www.scottreed.info/), [Cheng-Yang Fu](http://www.cs.unc.edu/~cyfu/), [Alexander C. Berg](http://acberg.com/). <Br>
**[Pages]** <Br>
	https://github.com/weiliu89/caffe/tree/ssd  <Br>
	https://github.com/balancap/SSD-Tensorflow  <Br>
**[Description]** <Br>
1) 感觉是YOLO和RPN的结合版, 又快又好型, 缺点是对小目标检测效果不佳
2) 将feature map划分成网格, 在default box上预测bounding box相对于default boxes的offset和每一类的score, 在后几个卷积层上进行划分和bbox预测，以实现多尺度检测, 用卷积代替YOLO的全连接.
3) 与YOLO相比, 共同点是直接将feature划分成网格, 并在default box上检测; 区别是: 1.使用了更多ratio和scale, 2.为每个得出的bbox分别预测每类的score而不是将检测object和分类分别进行,提高了精度; 3.用小kernel卷积代替了全连接层
4) 与RPN相比, 共同点是都定义了一系列anchor, 区别是: 1.初始框是预先划分好的网格而不是sliding window搜索的; 2. 在多个层的feature map上检测以实现multi-scale, 而不是在最后的feature map上定义多个尺度的anchor box

## Other Interesting Methods  


# Taditional Classical Methods


# Datasets

# Leaderboards
 PASCAL VOC http://host.robots.ox.ac.uk:8080/leaderboard/main_bootstrap.php <Br>
  ILSVRC2016 http://image-net.org/challenges/LSVRC/2016/results <Br>
# Sources-Lists
https://handong1587.github.io/deep_learning/2015/10/09/object-detection.html <Br>

