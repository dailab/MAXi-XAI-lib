# Example: Quantizer Implementation

![MAX Class Diagram](../../img/mael_class_diagram.png)

Let's assume we have an inference model which returns segmentation masks for a given dataset. The model assigns each pixel a value between 0 and 255. The higher the value, the more likely it belongs to a certain segment to be classified in the image.

In that case, the model will output a matrix with the same shape as the original image. In order to use the _CEM_ explanation model though, we have to somehow transform the prediction result into a class score vector, typically seen in classification problems. Furthermore, that transformation function has to be continuous and sensitive to changes in the image.

Suppose the inference results in the following prediction:

```python
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

The code below shows the **_BinaryConfidenceMethod_** class that converts and reduces our prediction to a vector of shape (2,) in range between -1.0 to 1.0. The first index of the vector represents the probability score of a class being absent in the image. The second index of the vector represents the probability score of a certain class being present.

**Note that it is explicitly recommended to inheret from the provided base class `BaseQuantizer` in order to easily comply with our API!**

```python
from maxi.lib.inference.quantizer.base_quantizer import BaseQuantizer

def calculate_confidence(segmentation_mask: np.ndarray, max_pixel_value: int = 255) -> float:
    return np.mean(segmentation_mask) / max_pixel_value

class BinaryConfidenceMethod(BaseQuantizer):
    def __init__(
        self,
        preprocess: Processor = quantizer_utils.identity,
        confidence_calculator: Callable[[np.ndarray], float] = calculate_confidence,
    ) -> None:
        """Binary Confidence Quantizer Method

        Description:
            This quantizer method takes an arbitrary prediction, calculates the confidence score and \
            constructs an array of binary classification format. \
            In order to extract the confidence, the user needs to provide a suitable ``confidence calculator`` \
            method.

        Output format:
            [-(confidence - 0.5) * 2, (confidence - 0.5) * 2]

        Args:
            preprocess (Processor, optional): Preprocessing procedure before quantizing.
                Defaults to identity function.
            confidence_calculator (Callable[[np.ndarray], float], optional): Method to calculate the confidence.
                Has to reduce the (preprocessed) prediction to a single value (1,).
                Defaults to calculate_confidence.
        """
        super().__init__(preprocess)
        self.confidence_calculator = confidence_calculator

    def __call__(self, prediction: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Performs the quantization

        Args:
            prediction (np.ndarray): Any type of inference result (e.g. a segmentation mask).

        Returns:
            np.ndarray: Prediction of binary classification format ([-(confidence - 0.5) * 2, (confidence - 0.5) * 2])
        """
        preprocessed_pred = super().__call__(prediction)
        # calculate confidence
        score = (self.confidence_calculator(preprocessed_pred) - 0.5) * 2
        return np.array([-score, score])
```

Let's see what that new quantizer produces for our prediction:

```python
>>> quantizer = BinaryConfidenceMethod()
>>> quantizer(prediction)
[ 0.09772549 -0.09772549]
```

Now, that we have written an appropriate quantizer method for our domain, we can pass the quantizer instance to the _**InferenceWrapper**_ and the latter to the _**ExplanationGenerator**_.

```python
    # initialize the InferenceWrapper with our inference method and the quantizer
    infer_wrapper = InferenceWrapper(
        inference_method,
        quantizer,
    )

    # parse the 'ExplanationGenerator.run' function the image to be explained as well as
    # the inference_wrapper in order to start the explanation procedure
    cem.run(image=image, inference_call=infer_wrapper)
```

[Back to main page](../../../README.md)
