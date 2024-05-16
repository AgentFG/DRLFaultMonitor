# 20240302
todynet在cartpole上的预测准确率非常高，可以用于selfcheck
很好用，即使是在step生成的情况下都是好使得
/home/cy/MTSC/TodyNet/model/03.02_gpu0_dyGIN2d_CartPole_exp.pth

# 20230303
todynet在Lunarlander上效果不好，初步分析情况在于，可能存在过拟合

# 20240304
todynet在BipedalWalkerHC上的效果很好，可以用于selfcheck，这是通过过采样之后实现的
/home/cy/MTSC/TodyNet/model/03.04_gpu0_dyGIN2d_NewBWHC1_exp.pth

如果我要用这个selfcheck做容错的工作的话，这个最多算是一个checker，但是如何进行recover呢？

# 20240312
todynet在BipedalWalkerHC上的，输入为`状态+动作+评分`，效果极佳，正确与否的评价更为准确了，说明`状态+动作+评分`的模式是有效的。
/home/cy/MTSC/TodyNet/model/03.12_gpu0_dyGIN2d_BWHCACVA_exp.pth


# 20240314
data/ucr数据集中所有数据如果包括动作+评分，则会带后缀ACVA，否则只有状态，不带后缀

NewBWHC1是数据增广之后

# 20240315
Walker2d上面的效果并不好，和lunarlander是一样的，只能识别不发生故障的序列，只要没见过的序列都会被定义为故障的，因此一开始就是1
