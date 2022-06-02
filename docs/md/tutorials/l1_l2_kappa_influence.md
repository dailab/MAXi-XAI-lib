# `l1`, `l2` & (CEM)`kappa` Influence on the Explanation Generation

For the provided optimizers, it might is essential to chose the right combination of hyperparameters - namely `l1`, `l2` and `kappa`. As you will see in this tutorial, the configuration of the latters heavily influences the result that is generated.

## Experiment Setup

```python
loss_class = maxi.loss.TF_CEMLoss
optimizer_class = maxi.optimizer.AdaExpGradOptimizer
gradient_class = maxi.gradient.TF_Gradient

loss_kwargs = {"mode": "PP", "c": 1, "gamma": 3, "K": 2}
optimizer_kwargs = {
    "l1": 0.05,
    "l2": 0.005,
    "channels_first": False,
}
gradient_kwargs = {"mu": None}
```

## kappa Influence

### Kappa = 2

![Kappa_2](./../../img/tutorials/l1_l2_kappa/kappa_2.png)

### Kappa = 10

![Kappa_10](./../../img/tutorials/l1_l2_kappa/kappa_10.png)

### Kappa = 20

![Kappa_20](./../../img/tutorials/l1_l2_kappa/kappa_20.png)

## l1 Influence

### l1 = 0.000005

![Kappa_2](./../../img/tutorials/l1_l2_kappa/l1_0_000005.png)

### l1 = 0.005

![Kappa_10](./../../img/tutorials/l1_l2_kappa/l1_0_005.png)

### l1 = 5.0

![Kappa_20](./../../img/tutorials/l1_l2_kappa/l1_5_0.png)

## l2 Influence

### l2 = 0.0000005

![Kappa_2](./../../img/tutorials/l1_l2_kappa/l2_0_0000005.png)

### l1 = 0.005

![Kappa_10](./../../img/tutorials/l1_l2_kappa/l2_0_005.png)

### l1 = 5.0

![Kappa_20](./../../img/tutorials/l1_l2_kappa/l2_5_0.png)
