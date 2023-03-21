# stochastic gradient descent

When should we use the stochastic gradient descent?

* If we have tons of data/predictors. (regular gradient descent might not be computationally feasible(slow)
* is useful, if there are redundancies in the data
* it’s more easy if we have new data, and we can easily to get another step

How it works?

* stochastic gradient descent will first randomly pick a point for each step.
* use the one/mini-batch/small subset of sample to calculate the derivatives.

Note.

* it’s sensitive to the learning rate. (same as gradient descent)
* **Schedule** : schedule is the strategy of learning rate is start from the large and then make it smaller with each step
* If fail to converge on parameter estimates, try futzing with this setting

![Gradient Descent in Neural Network – Study Machine Learning](https://studymachinelearning.com/wp-content/uploads/2019/10/sgd-GD.png)

![Brief of the Stochastic Gradient Descent | Neural Network Calculation](https://www.xenonstack.com/hubfs/stochastic-gradient-descent.png)

### **隨機梯度下降法(Stochastic gradient descent, SGD)**

我們一般看深度學習的介紹，最常看到的最佳化名稱稱為「 **隨機梯度下降法(Stochastic gradient descent, SGD)** 」(這篇我為了縮短篇幅，Mini-batch SGD我把它歸納到SGD內)，我們簡單說一下這個跟GD差在哪邊。

在更新參數的時候GD我們是一次用全部訓練集的數據去計算損失函數的梯度就更新一次參數。SGD就是一次跑一個樣本或是小批次(mini-batch)樣本然後算出一次梯度或是小批次梯度的平均後就更新一次，那這個樣本或是小批次的樣本是隨機抽取的，所以才會稱為**隨機**梯度下降法。
