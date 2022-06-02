## Example: _Optimizer_ Implementation

![MAX Class Diagram](../../img/mael_class_diagram.png)

For this example, we will walk through the implementation of the **_Adaptive Optimistic Exponentiated Gradient Optimizer_** into our library.

**Note that it is explicitly recommended to inheret from the provided base class `BaseOptimizer` in order to comply with our API!**

Suppose we have the following implementation of the _Adaptive Optimistic Exponentiated Gradient Optimizer_:

<details open>
<summary>Click here</summary>

```python
class AOExpGrad(object):
    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        func_p: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        l1: float = 1.0,
        l2: float = 1.0,
        eta: float = 1.0,
    ):
        self.func = func
        self.func_p = func_p
        self.x = np.zeros(shape=x0.shape)
        self.x[:] = x0
        self.y = np.zeros(shape=x0.shape)
        self.y[:] = x0
        self.d = self.x.size
        self.lower = lower
        self.upper = upper
        self.eta = eta
        self.lam = 0.0
        self.t = 0.0
        self.beta = 0
        self.l1 = l1
        self.l2 = l2
        self.h = np.zeros(shape=self.x.shape)

    def update(self) -> np.ndarray:
        self.t += 1.0
        self.beta += self.t
        g = self.func_p(self.y)
        self.step(g)
        self.h[:] = g
        return self.y

    def step(self, g):
        self.update_parameters(g)
        self.md(g)

    def update_parameters(self, g):
        self.lam += (self.t * lina.norm((g - self.h).flatten(), ord=np.inf)) ** 2

    def md(self, g):
        beta = 1.0 / self.d
        alpha = np.sqrt(self.lam) / np.sqrt(np.log(self.d)) * self.eta
        if alpha == 0.0:
            alpha += 1e-6
        z = (np.log(np.abs(self.x) / beta + 1.0)) * np.sign(self.x) - (
            self.t * g - self.t * self.h + (self.t + 1) * g
        ) / alpha
        x_sgn = np.sign(z)
        if self.l2 == 0.0:
            x_val = beta * np.exp(np.maximum(np.abs(z) - self.l1 * self.t / alpha, 0.0)) - beta
        else:
            a = beta
            b = self.l2 * self.t / alpha
            c = np.minimum(self.l1 * self.t / alpha - np.abs(z), 0.0)
            abc = -c + np.log(a * b) + a * b
            x_val = (
                np.where(
                    abc >= 15.0,
                    np.log(abc) - np.log(np.log(abc)) + np.log(np.log(abc)) / np.log(abc),
                    lambertw(np.exp(abc), k=0).real,
                )
                / b
                - a
            )
            # x_val = lambertw(a * b * np.exp(a * b - c), k=0).real / b - a
        y = x_sgn * x_val
        self.x = np.clip(y, self.lower, self.upper)
        self.y = (self.t / self.beta) * self.x + ((self.beta - self.t) / self.beta) * self.y
```

</details>

As the next step to enable the utilization of the optimizer within our library, we construct a new optimizer class that inherits from the `BaseOptimizer` class. A more detailed documentation of the class can be found here: [BaseOptimizer Documentation](https://tuananhroman.github.io/empaia_max_pydoc/lib/computation_components/optimizer/base_optimizer.html)

```python
from maxi.lib.computation_components.optimizer import BaseOptimizer

class AoExpGradOptimizer(BaseOptimizer):
    def __init__(
        self,
        loss: BaseExplanationModel,
        gradient: BaseGradient,
        org_img: np.ndarray,
        x0: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        eta: float = 1.0,
        l1: float = 0.5,
        l2: float = 0.5,
        *args,
        **kwargs
    ):
        # required attributes of the optimizer class
        super().__init__(
            loss,
            gradient,
            org_img,
            x0,
            lower,
            upper,
        )

        self.eta, self.l1, self.l2 = eta, l1, l2

        # here we initialize the algorithm class from above
        self.alg = AOExpGrad(
            func=self.loss,
            func_p=self.gradient,
            x0=self.x0,
            lower=self.lower,
            upper=self.upper,
            eta=self.eta,
            l1=self.l1,
            l2=self.l2,
        )

        self.call_count = 0
```

The API requires one to implement the `step()` function where an optimization update is executed. As is hinted by the documentation, the function needs to return an `OptimizeResult` instance which encapsulates optimization related information as the current optimization result, the loss and the iteration counter.

`OptimizeResult` can be imported via `from scipy.optimize import OptimizeResult`. A more detailed documentation can be found here: [official scipy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html).

```python
    def step(self, *args, **kwargs) -> OptimizeResult:
        # optimizer step, with y being the intermediate result
        y = self.alg.update() if self.call_count != 0 else self.x0

        # increment the counter
        self.call_count = self.call_count + 1

        # init the OptimizeResult
        return OptimizeResult(
            func=self.alg.func(y),
            x=y,
            nit=self.call_count,
            nfev=self.call_count,
            success=(y is not None),
        )
```

The ExplanationGenerator will parse the loss class, gradient calculation class, the target image, the starting point of the optimization amd lower as well as upper bound during component initialization to the optimizer class. Generally, the base classes' attributes will be parsed by the `ExplanationGenerator`.
Thus, the user only has to specify `l1`, `l2`, `eta` as keyword arguments for the `ExplanationGenerator` as can be seen in the example below:

```python
     # chose desired component classes for the loss, optimizer and gradient
     # set our 'Adaptive Optimistic Exponentiated Gradient Optimizer' as optimization algorithm
    loss_class = maxi.TF_CEMLoss
    optimizer_class = maxi.AoExpGradOptimizer
    gradient_class = maxi.TF_Gradient

    # specify the configuration for the components
    loss_kwargs = {"mode": "PP", "c": 100, "gamma": 300, "K": 30, "AE": AE}
    optimizer_kwargs = {"l1": 0.1, "l2": 30, "eta": 1.0}
    gradient_kwargs = {}

    # instantiate the "ExplanationGenerator" with our settings
    cem = maxi.ExplanationGenerator(
        loss=loss_class,
        optimizer=optimizer_class,
        gradient=gradient_class,
        loss_kwargs=loss_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        gradient_kwargs=gradient_kwargs,
        num_iter=1500,
        save_freq=500,
    )
```

[Back to main page](../../../README.md)
