# Supervised Learning Algorithm

Supervised learning is a type of machine learning which the algorithm learns from labeled data, helps you to predict outcomes for unforeseen data. The goal is for the algorithm to learn the relationship between the input data and the labeled output data, so that it can accurately predict the output for new, unseen input data. Examples of supervised learning include classification problems, such as image recognition, and regression problems, such as predicting housing prices.It means some data is already tagged with the correct answer and seems like in the presence of a supervisor or a teache

## Regularization

Motivation for [regularization]():

1. Model with too many parameters will overly complex and might overfit
2. Outliers can skew line and cause bad generalization
3. Data with too many features can get extreme coefficients

> *Regularization, in whatever form you use it, is mainly about **reducing complexity and overfitting** so that your model generalizes well to new unseen data! *

Extreme coefficients are unlikely to yield good model generalization, so what we can do is simply constrain the size of the model coefficients.

---

### L1 (Lasso) Regularization

L1 regularization adds the sum of the absolute value of the coefficient to loss function which help reduce model complexity and improve generality. The L1 regularized coefficients located on the diamond-shaped zone has below pros:

1. L1 regularization allows superfluous coefficients to shrink directly to zero

   ![1680476148833](image/README/1680476148833.png)
2. L1 regularization can reduce the number of features and select features

   ![1680476406729](image/README/1680476406729.png)

```python
class LinearRegression:  # NO MODIFICATION NECESSARY
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                          loss_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter)

```

### L2 (Ridge) Regularization

L2 regularization adds the sum of the square of the parameters into the loss function which also help reduce model complexity and improve generality. The L2 regularized coefficients located on the circle-shaped zone has below pros:

1. L2 regularization tends to shrink coefficients evenly

   ![1680477694529](image/README/1680477694529.png)
2. L2 regularization useful when you have collinear/codependent features since L2 regularization will reduce the variance of these coefficient estimates

   ![1680477777450](image/README/1680477777450.png)

```python
class RidgeRegression:  
    def __init__(self,
                 eta=0.00001, lmbda=0.0, addB0=False,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.addB0 = False

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        B0 = np.mean(y)
        self.B = minimize(X, y, loss_gradient_ridge,
                          self.eta, self.lmbda,
                          self.max_iter, addB0=False,
                          precision=1e-9)
        self.B = np.vstack([B0, self.B])

```

### Elastic Net Regularization

You may be wondering if you can combine both L1 and L2 penalties into a single model. The answer is yes, and it is called Elastic Net. But it not common to use.

![1680478033583](image/README/1680478033583.png)

![1680478053716](image/README/1680478053716.png)

#### Key Conclusion for Regularization

1. Regularization adds a little bit of bias to the model in order to reduce variance, i.e. increase generality by restricting the size of the 𝛽
2. Hard constraint: min-loss is inside safe zone or on zone border
3. Soft constraint: penalty discourages bigger parameters
4. L1(Lasso) shrinks coefficients to 0, so it can be used for feature selection
   L2(Ridge) shrinks coefficients towards each other and towards 0, but not to 0
5. With correlated features, L1 will pick one at random, L2 reduces the variance of β**i which reduces impact of collinearity
6. L1 linear regression and L1/L2 logistic regression require iterative solution

---

## Gradient Descent

What options do we have for finding β that minimizes L?

• we could do a grid search over a large grid of parameter values (only for models that have a very small number of parameters)

• choose randomly?

• Newton’s method?

#### Slope(gradient) & Learning rate(η)

To determine which direction to adjust β, we use the slope (gradient) of the loss function at the current value of β. The gradient, specifically the sign of the gradient, tells us which direction we need to move in order to keep going **downhill** towards a minimum.

And the gradient is the derivative and the sigh of derivative tell us the direction: if > 0, go negative; if < 0, go positive; if = 0, stop

![1680479872824](image/README/1680479872824.png)

If the slope is negative, then we want to move to the right, because that would mean we are moving downhill. If the slope is positive, we want to move to the left so that we don’t keep going uphill. So, we want to move in the direction of the negative derivative. Note also that the derivative has a magnitude which tells us how steep the slope is in that direction. We can use that information to tell us how big of a step we should take (how much to adjust β).

We want to be able to control our steps, so we will apply what’s called a learning rate (η) which lets us control our step size. Using η and the slope (derivative), the algorithm is:

![1680480192858](image/README/1680480192858.png) *where t indicates which step we are on.*

However, if the learning rate is to big, the step size will too large and bounce back and forth.

![1680480506273](image/README/1680480506273.png)

pseudocode:

> b = random value
> while not_converged:
> b = b - rate * gradient(b)

```python
# Define your loss function and gradient function
def f(b): return (b-2)**2
def gradient(b): return 2*(b-2)
# pick a random value of b to start
# choose a learning rate
b = np.random.uniform(0,4)
rate = 0.2
# loop
for t in range(10):
b = b - rate * gradient(b)
```

#### Momentum

Add a fraction of previous step to current step in order to reinforce movement in the same direction as previous step.

![1680480932215](image/README/1680480932215.png)

Here, γ is a hyperparameter that controls the momentum that we add.

#### Adagrad

Adapt learning rate η to each of parameters β, so that taking different steps in each direction. A single learning rate might have us bounce back and forth or move too slowly down the shallow slope.

Each parameter's learning rate η is dividing by the square root of the sum of the squared historical gradients, i.e. we save the history of the gradients, square these, sum them, and take the square root.

![1680481310830](image/README/1680481310830.png)

*ϵ is just a smoothing paramater to keep from dividing by zero, and can be set very small.*
