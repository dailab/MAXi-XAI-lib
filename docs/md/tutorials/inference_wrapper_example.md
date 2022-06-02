# Example: InferenceWrapper Usage

The [_Inference Wrapper_](https://tuananhroman.github.io/empaia_max_pydoc/lib/inference/inference_wrapper.html) essentially interconnects the inference method with the [_Inference Quantizer_](https://tuananhroman.github.io/empaia_max_pydoc/lib/inference/quantizer/index.html) (and optional preprocessing procedure). It is especially useful when your inference model's prediction format is not compatible with the _Explanation Model_. In this example, a setup where this component proves necessary is presented.

Let's assume we have an inference model which returns segmentation masks for a given dataset. The model assigns each pixel a value between 0 and 255. The higher the value, the more likely it belongs to a certain segment to be classified in the image.

In that case, the model will output a matrix with the same shape as the original image. In order to use the _CEM_ explanation model though, we have to somehow transform the prediction result into a class score vector, typically seen in classification problems. Furthermore, that transformation function has to be continuous and sensitive to changes in the image.
We will consider the `BinaryConfidenceMethod` as _InferenceQuantizer_ ([BinaryConfidenceMethod documentation](https://tuananhroman.github.io/empaia_max_pydoc/lib/inference/quantizer/confidence_method.html)) which will transform our segmentation mask into a binary classification representation.

The first index of the vector represents the probability score of a class being absent in the image. The second index of the vector represents the probability score of a certain class being present.

- The prediction output

```python
>>> prediction = predict(image)
>>> prediction
[[164 112 225 253 208 156  52  65 199 126]
 [211  17 213 122  71 188 102  97 159   3]
 [147 190   2  10 164  13  71  40 239  47]
 [ 77  42   8  67  57 180 101  40  30  59]
 [ 72 188  15 135  34  91 159 208 109 153]
 [ 66 237 248 248 225  40 118  85  92  64]
 [ 86  56  50 119  62 162  11 247 239  78]
 [178  67 127 119  58 183  85 146 245 147]
 [  5  85 144  50 233  73 118  74 191 220]
 [ 42 159  51  79  15  70 118  15 135 248]]
```

- After quantization

```python
>>> quantizer = BinaryConfidenceMethod()
>>> quantizer(prediction)
[ 0.09772549 -0.09772549]
```

Now that we have setup the quantizer, we have to connect the entities through the `InferenceWrapper`. The resulting instance will then produce the desired output format as prediction.

```python
>>> inference_wrapper = maxi.InferenceWrapper(inference_model=predict, quantizer=quantizer)
>>> inference_wrapper(image)
[ 0.09772549 -0.09772549]
```

Eventually, we can parse the explanation method conforming `InferenceWrapper` to the `ExplanationGenerator.run()` method.

```python
# 'cem' is an instance of the 'ExplanationGenerator'
>>> result = cem.run(image=image, inference_call=inference_wrapper)
```
