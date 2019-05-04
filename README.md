# Personal-Object-Detection-Paper-Record
# Under construction!
# Table of Contents
- [Deep Learning Methods](#deep-learning-methods)
  - [Two Stages](#two-stages)
  - [One Stage](#one-stage)
  - [Other Interesting Methods](#other-interesting-methods)
- [Taditional Classical Methods](#traditional-classical-methods)
- [Datasets](#datasets)
- [Leaderboards](#leaderboards)
- [Sources-Lists](#sources-lists)

# Rank
- Semantic Segmentation<Br>
	- ★★★ <Br>
		**[R-CNN]**, **[Faster R-CNN]**, **[YOLO]**, **[SSD]** <Br>
	- ★★  <Br>
		**[SPPNet]**, **[Fast R-CNN]**, **[R-FCN]**, **[RPN]**, **[YOLOv3]**, **[CornerNet]** <Br>
	- ★  <Br>
		**[YOLO9000]**, **[Objects as Points]**, **[Keypoint Triplets]** <Br>
	- ♥  <Br>
<Br>
	
# Deep Learning Methods
## Two Stages

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

### **R-FCN ★★**
**[Paper]**   R-FCN:Object Detection via Region-based Fully Convolutional Networks <Br>
**[Year]** NIPS 2016 <Br>
**[Authors]** 	[Jifeng Dai](http://www.jifengdai.org/), [Yi Li](https://liyi14.github.io/), [Kaiming He](http://kaiminghe.com/), [Jian Sun](http://www.jiansun.org/) <Br> 
**[Pages]**  https://github.com/daijifeng001/R-FCN  <Br>
**[Description]** <Br>
1) 从提高检测和分类共享计算出发, 提出了一种基于position-sensitive score map的目标检测方法, 权衡在图像分类中的平移不变性和在物体检测中的平移变换性这样一种两难境地. 在速度和性能上取得了不错的平衡. <Br>
2) 经过CNN特征提取(ResNet-101)得到feature map, 此feature map分别作为RPN和位置敏感分类两支的输入. 分类一支生成K*K*(C+1)个channel的feature map， K*K个节点分别对应K*K网格的不同位置, 如K=3时分别代表某个类别左上至右下9个位置对应的score, 这样输出就encode了各类别的相对位置信息. 将RPN得到的ROI进行position-sensitive ROI pooling, 得到K*K个ROI的得分网格, 再对这一网格进行投票即可得到该ROI的类别. <Br>
3) R-FCN不存在Faster R-CNN中那样的region-wise layers, 速度大大提升, 且性能很好. 至于其中的道理嘛... 之后可仔细阅读论文和代码. <Br>

### **FPN ★★**
**[Paper]**  Feature Pyramid Networks for Object Detection <Br>
**[Year]** NIPS 2016 <Br>
**[Authors]** 	[Tsung-Yi Lin](https://vision.cornell.edu/se3/people/tsung-yi-lin/), [Piotr Doll´ar](http://pdollar.github.io/), [Ross Girshick](http://www.rossgirshick.info/), [Kaiming He](http://kaiminghe.com/), [Bharath Hariharan](http://home.bharathh.info/), [Serge Belongie](http://blogs.cornell.edu/techfaculty/serge-belongie/) <Br> 
**[Pages]**    <Br>
**[Description]** <Br>
1) 仿照SSD设计了多尺度特征融合金字塔, 在应用多尺度信息提升精度(特别是小目标精度)的同时, 保证了一定的速度. FPN是一种通用的feature extractor, 可用于目标检测的许多地方. <Br>
2) FPN结构类似于U-Net, 分为bottom-up和top-down两步, top-down的每个stage用前一阶段的特征和对应的bottom-up特征融合而成, 每个stage融合后的特征都用于预测, 以检测不同scale的目标. <Br>

**[Reference]**  <Br>
https://blog.csdn.net/u014380165/article/details/72890275 <Br>
https://blog.csdn.net/jesse_mx/article/details/54588085 <Br>

### **Soft-NMS**
**[Paper]**  Soft-NMS -- Improving Object Detection With One Line of Code <Br>
**[Year]** ICCV 2017<Br>
**[Authors]** Navaneeth Bodla, [Bharat Singh](https://bharatsingh.net/), [Rama Chellappa](http://users.umiacs.umd.edu/~rama/), [Larry S. Davis](http://users.umiacs.umd.edu/~lsd/)<Br> 
**[Pages]** https://github.com/bharatsingh430/soft-nms   <Br>
**[Description]** <Br>
	
### **CoupleNet**
**[Paper]**  CoupleNet: Coupling Global Structure with Local Parts for Object Detection <Br>
**[Year]** ICCV 2017<Br>
**[Authors]** Yousong Zhu, Chaoyang Zhao, Jinqiao Wang, Xu Zhao, Yi Wu, Hanqing Lu <Br> 
**[Pages]** https://github.com/tshizys/CoupleNet   <Br>
**[Description]** <Br>

### **Relation Networks**
**[Paper]** Relation Networks for Object Detection <Br>
**[Year]** CVPR 2018 Oral<Br>
**[Authors]** [Han Hu], [Jiayuan Gu](https://sites.google.com/eng.ucsd.edu/jiayuan-gu), Zheng Zhang, [Jifeng Dai](http://www.jifengdai.org/), Yichen Wei <Br> 
**[Pages]** https://github.com/msracver/Relation-Networks-for-Object-Detection   <Br>
**[Description]** <Br>

### **R-FCN-3000**
**[Paper]**  R-FCN-3000 at 30fps: Decoupling Detection and Classification <Br>
**[Year]** CVPR 2018<Br>
**[Authors]** [Bharat Singh](https://bharatsingh.net/), Hengduo Li, [Abhishek Sharma](https://scholar.google.com/citations?user=18fTep8AAAAJ&hl=en&oi=sra), [Larry S. Davis](http://users.umiacs.umd.edu/~lsd/) <Br> 
**[Pages]** https://github.com/mahyarnajibi/SNIPER/tree/cvpr3k   <Br>
**[Description]** <Br>

### **MegDet**
**[Paper]**  MegDet: A Large Mini-Batch Object Detector <Br>
**[Year]** CVPR 2018<Br>
**[Authors]** [Chao Peng](http://www.pengchao.org/), [Tete Xiao](http://tetexiao.com/), [Zeming Li](http://www.zemingli.com/), [Yuning Jiang](https://yuningjiang.github.io/), [Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ&hl=en), Kai Jia, [Gang Yu](http://www.skicyyu.org/), [Jian Sun](http://www.jiansun.org/) <Br> 
**[Pages]** https://github.com/mahyarnajibi/SNIPER/tree/cvpr3k   <Br>
**[Description]** <Br>

### **SNIP**
**[Paper]**  An analysis of scale invariance in object detection-snip <Br>
**[Year]** CVPR 2018 Oral<Br>
**[Authors]** [Bharat Singh](https://bharatsingh.net/), [Larry S. Davis](http://users.umiacs.umd.edu/~lsd/) <Br> 
**[Pages]** https://github.com/mahyarnajibi/SNIPER   <Br>
**[Description]** <Br>

### **SNIPER**
**[Paper]**  SNIPER: Efficient Multi-Scale Training <Br>
**[Year]** NIPS 2018 <Br>
**[Authors]** [Bharat Singh](https://bharatsingh.net/),[Mahyar Najibi](http://users.umiacs.umd.edu/~najibi/), [Larry S. Davis](http://users.umiacs.umd.edu/~lsd/) <Br> 
**[Pages]** https://github.com/mahyarnajibi/SNIPER   <Br>
**[Description]** <Br>

### **Cascade R-CNN**
**[Paper]**  Cascade R-CNN: Delving into High Quality Object Detection <Br>
**[Year]** CVPR 2018 <Br>
**[Authors]** [Zhaowei Cai](https://sites.google.com/site/zhaoweicai1989/), [Nuno Vasconcelos](http://www.svcl.ucsd.edu/~nuno/) <Br> 
**[Pages]** https://github.com/zhaoweicai/cascade-rcnn   <Br>
**[Description]** <Br>

### **RefineDet**
**[Paper]** Single-Shot Refinement Neural Network for Object Detection <Br>
**[Year]** CVPR 2018 <Br>
**[Authors]**  [Shifeng Zhang](http://www.cbsr.ia.ac.cn/users/sfzhang/), [Longyin Wen](http://www.cbsr.ia.ac.cn/users/lywen/), [Xiao Bian](https://sites.google.com/site/cvbian/), [Zhen Lei](http://www.cbsr.ia.ac.cn/users/zlei/), [Stan Z. Li](http://www.cbsr.ia.ac.cn/users/szli/) <Br> 
**[Pages]** https://github.com/sfzhang15/RefineDet  <Br>
**[Description]** <Br>


### **IoUNet**
**[Paper]**  Acquisition of Localization Confidence for Accurate Object Detection<Br>
**[Year]** ECCV 2018 Oral<Br>
**[Authors]** Borui Jiang, Ruixuan Luo, [Jiayuan Mao](http://vccy.xyz/), [Tete Xiao](http://tetexiao.com/), [Yuning Jiang](https://yuningjiang.github.io/)  <Br> 
**[Pages]** https://github.com/vacancy/PreciseRoIPooling  (Uncomplete) <Br>
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

### **YOLO9000 ★☆**
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

### **YOLOv3 ★★**
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

### **Focal Loss**
**[Paper]** Focal Loss for Dense Object Detection <Br>
**[Year]** ICCV 2017 Oral<Br>
**[Authors]** [Tsung-Yi Lin](https://vision.cornell.edu/se3/people/tsung-yi-lin/), [Priya Goyal](https://scholar.google.com/citations?user=-9yiQMsAAAAJ&hl=en&oi=ao), [Ross Girshick](http://www.rossgirshick.info/), [Kaiming He](http://kaiminghe.com/), [Piotr Dollar](https://pdollar.github.io/) <Br> 
**[Pages]**    <Br>
**[Description]** <Br>

### **CornerNet ★★**
**[Paper]**  CornerNet: Detecting Objects as Paired Keypoints <Br>
**[Year]** ECCV 2018 Oral<Br>
**[Authors]** [Hei Law](https://heilaw.github.io/), [Jia Deng](https://www.cs.princeton.edu/~jiadeng/) <Br> 
**[Pages]** https://github.com/umich-vl/CornerNet   <Br>
**[Description]** <Br>
1) Anchor-free目标检测的代表性算法, 与一般的以目标为中心预测bounding box的思路不同, 提出了预测top-left和bottom-right角点进而得到目标框的目的. 该方法分别计算左上和右下角点的heatmap和embedding, 最后根据embedding向量的距离找到同一目标对应的一对top-left和bottom-right角点. 整体做法与人体关节点检测颇有相似之处. <Br>
2) 提出了corner pooling, 目的是更好地捕获水平和竖直方向的信息, 经过corner pooling后, 分别得到heatmap, embedding和offset. offset是为了修正下采样过程中造成的位置偏移. embedding是为了将同一目标的两个角点group起来设置的, 设计了push和pull两个loss, 目的是使不同目标角点的embedding距离大, 同一目标角点的embedding小. 在计算loss时, 以focal loss为基础, 根据预测值与真值距离的大小使用不同权重<Br>
3) 性能上超过了现有的one stage方法, 与two stage的方法差距不大. 不过还没测过运行速度怎样.<Br>


### ** Objects as Points ★☆**
**[Paper]**  CenterNet :Objects as Points <Br>
**[Year]** arXiv 2019<Br>
**[Authors]** [Xingyi Zhou](http://xingyizhou.xyz/), [Dequan Wang](https://dequan.wang/), [Philipp Krähenbühl](http://www.philkr.net/) <Br> 
**[Pages]** https://github.com/xingyizhou/CenterNet   <Br>
**[Description]** <Br>
1) Anchor-free的目标检测算法, 使用基于heapmap的关键点检测方法检测目标的center point. 思路简洁, 可扩展到姿态估计, 3d目标检测等任务中, 在速度和性能上达到了很好的平衡. <Br>
2) 模型分为三个分支, i>基于heapmap的中心点检测, 因为取得是每个位置的最大值, 所以能起到NMS的作用, 因此可省去非end-to-end的NMS部分; ii>不区分类别的offset分支修正降采样带来的error; iii>不区分类别的目标尺寸预测分支. 另外还可以加入其他分支做3d, 姿态估计等. <Br>

### ** Keypoint Triplets ★☆**
**[Paper]**  CenterNet: Keypoint Triplets for Object Detection <Br>
**[Year]** arXiv 2019<Br>
**[Authors]** Kaiwen Duan, [Song Bai](http://songbai.site/), [Lingxi Xie](http://lingxixie.com/Home.html), [Honggang Qi](http://people.ucas.ac.cn/~hgqi), [Qingming Huang](https://scholar.google.com/citations?user=J1vMnRgAAAAJ&hl=zh-CN), [Qi Tian](https://scholar.google.com/citations?user=61b6eYkAAAAJ&hl=zh-CN)<Br> 
**[Pages]** https://github.com/Duankaiwen/CenterNet   <Br>
**[Description]** <Br>
1) 基于CornerNet提出的改进, 明显提升了CornetNet的性能 <Br>
2) 为了解决CornerNet的False Discovery问题, 提出了检测中心点, 推断时通过判断corner点围成的区域里是否有置信度高于一定阈值的中心点, 来判断是否保留该bounding box. <Br>
3) 在corner pooling的基础上提出了center pooling和cascade corner pooling, 更好的挖掘信息. <Br>
4) 实验及代码没研究 <Br>
	
## Other Interesting Methods  


# Taditional Classical Methods


# Datasets

# Leaderboards
 PASCAL VOC http://host.robots.ox.ac.uk:8080/leaderboard/main_bootstrap.php <Br>
  ILSVRC2016 http://image-net.org/challenges/LSVRC/2016/results <Br>
MSCOCO http://cocodataset.org/#detection-leaderboard <Br>
# Sources-Lists
https://handong1587.github.io/deep_learning/2015/10/09/object-detection.html <Br>

