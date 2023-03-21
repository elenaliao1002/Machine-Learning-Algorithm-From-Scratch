## XGBoost

### *1.1 XGB - Regression*

##### 1.1.1 The initial prediction

The predication could be anything, but by defualt it's 0.5 for regression and classification. 

##### 1.1.2 Calculate Similarity Score

Similarity = Sum of Residual, square / (# residuals + lambda)

> lambda is a regularization parameter.
>
> - lambda > 0 : it will penalty more on the small # residuals (inverse proportional to the # of residuals)

* smaller similarity: residuals are quite different in a node(they will cancel out each other)
* larger similarity: residuals are similar(do not cancel out) or just one residual

##### 1.1.3 Calculate Gain to evaluate different thresholds

Gain : compute how much better improve from the root to leaf nodes

看多分一層leave node 可以增加多少相似度

Gain = Similarity_left + Similarity_right - Similarity_root

##### 1.1.4 Pruning XGBoost Tree

- Gamma : 最下面那個branch的gain至少要超過gamma設定值(從下面開始檢查-prune)

  - calculate the difference between the gain assoicated with the lowest branch and the gamma.
  - if the gain > gamma -> don't remove branch
    - 代表該branch 可以增加相似度
  - if the gain < gamma -> remove the branch
  - Note : even we set gamma = 0 , we can still prune the tree (need to improve than previous nodes)
- lambda : bigger lambda, smaller gain -> more eaily to prune
