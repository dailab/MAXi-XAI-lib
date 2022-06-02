# Example: Explanation Model Implementation

![MAX Class Diagram](../../img/mael_class_diagram.png)

In this tutorial, we will present an example of how one would implement a custom loss function module into the MAX library. The `BaseExplanationModel` plays an integral part in the optimization process as it poses as target function of the explanation (optimization problem).

**Note that it is explicitly recommended to inheret from the provided base class `BaseExplanationModel` in order to comply with our API!**

[BaseExplanationModel Documentation](https://tuananhroman.github.io/empaia_max_pydoc/lib/loss/base_explanation_model.html)

Similar to the other components, the base class displays the obligatory interface that our new explanation model has to follow. Beside the `get_loss` method that every loss function must implement, the following attributes have to be parsed and set: `x0_generator`, `lower`, `upper`. These attributes may differ between explanation model formulations and have to be chosen carefully.

An example implementation can be found here: [CEMLoss Documentation](https://tuananhroman.github.io/empaia_max_pydoc/lib/loss/cem_loss.html)

```python
    # chose desired component classes for the loss, optimizer and gradient
    # set our 'Contrastive Explanation Method' as optimization algorithm
    loss_class = maxi.CEMLoss
    optimizer_class = maxi.AoExpGradOptimizer
    gradient_class = maxi.URVGradientEstimator

    # specify the configuration for the components
    # set the arguments for our loss method, we chose to generate the pertinent positive
    loss_kwargs = {"mode": "PP", "c": 100, "gamma": 300, "K": 30, "AE": AE}
    optimizer_kwargs = {"l1": 0.1, "l2": 30, "eta": 1.0}
    gradient_kwargs = {"mu": None, "num_iter": 150}

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
