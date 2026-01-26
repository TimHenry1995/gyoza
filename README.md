## Description
This repository provides code to build invertible disentangling flow models that can help to reshape complicated high-dimensional data-clouds (manifolds) into simpler, often low-dimensional manifolds which can be visualized more easily. The below diagram shows a simple example of a 2-dimensional manifold in the left panel. This manifold is an arc and its points can be described based on their **position** on the arc as well as the deviation from from, also known as **residual** variation. Using a flow model, space is remapped (see right panel) such that the position along the manifold aligns only with the vertical axis and the deviation (i.e. residual variation) is moved to the horizontal axis. Here, the input and output space of the model only have 2-dimensions, but in practice one can choose much higher dimensionalities. An n-dimensional manifold could thus be transformed such that position along the manifold is captured entirely by a small number k of dimensions (e.g. 1,2 or 3 for visualization) while the residual variance is moved to the remaining n-k axes. 

![Image](https://github.com/TimHenry1995/gyoza/blob/main/docs/Inference%20on%20Arc.png?raw=true)

Apart from visualization, it is also possible to map any changes made in the projected space back onto the original manifold. The below animation shows how one can slide along the position axis of the model's output and thanks to the fact that this type of flow model is trivially invertible, map the changes back onto the original manifold. 

![til](https://github.com/TimHenry1995/gyoza/blob/main/docs/Inverse%20Model%20on%20Arc.gif?raw=true)

This has applications in interpreting and systematcially manipulating latent spaces of artificial neural networks. It is, for instance, possible to pass a stimulus half-way through an artificial neural network to get its latent representation and then disentangle the manifold according to human-interpretable underlying factors whose variation is moved to distinct dimensions. Manipulatioins in the disentangled space can then be re-mapped into the original manifold and the artificial neural network's downstream processing can be continued. This allows for experiments arguing about the causal role of said human-interpretable factors in the artificial neural network's inference.

## Installation
```
pip install gyoza
```

## Documentation
Detailed documentation can be found on the companion website of [read-the-docs](https://gyoza.readthedocs.io/en/latest/)

## Tutorial
A tutorial can be found in the [tutorials](https://github.com/TimHenry1995/gyoza/tree/main/tutorials) folder of the Github repository.

## References
The main resources used to write the code of this toolbox are
- Dinh, L., Sohl-Dickstein, J. & Bengio, S. (2016). “Density estimation using Real NVP” [arXiv:1605.08803](https://arxiv.org/abs/1605.08803) 
- Kingma, D. P. & Dhariwal, P. (2018) “Glow: Generative Flow with Invertible 1x1 Convolutions” [arXiv:1807.03039](https://arxiv.org/abs/1807.03039)
- Dinh, L., Krueger, D. & Bengio, Y. (2015) “NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION” [arXiv:1410.8516](https://arxiv.org/abs/1410.8516)
- Sankar, A., Keicher, M., Eisawy, R., Parida, A., Pfister, F., Kim, S., T. & Navab, N. (2021) “GLOWin: A Flow-based Invertible Generative Framework for Learning Disentangled Feature Representations in Medical Images” [arXiv:2103.10868](https://arxiv.org/abs/2103.10868)
- Meng, C., Song, Y., Song, J. & Ermon, S. (2020) “Gaussianization Flows” [arXiv:2003.01941](https://arxiv.org/abs/2003.01941)
- Esser, P., Rombach, R., & Ommer, B. (2020). “A Disentangling Invertible Interpretation Network for Explaining Latent Representations.” [arXiv:2004.13166](https://arxiv.org/abs/2004.13166)

















