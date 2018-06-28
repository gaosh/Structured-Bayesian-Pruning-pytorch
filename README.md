# Structured-Bayesian-Pruning-pytorch
pytorch implementation of [Structured Bayesian Pruning, NIPS17](https://arxiv.org/pdf/1705.07283.pdf). Authors of this paper provided [TensorFlow implementation](https://github.com/necludov/group-sparsity-sbp).

Some preliminary results on MNIST:

| Network | Method  | Error |	Neurons per Layer |
|--- | --- | --- | --- |
| LeNet-5 | Orginal | 0.68 |20 - 50 - 800 - 500|
|         | SBP     | 0.86 |3 - 18 - 284 - 283|
|         | SBP*    | 1.17 |8 - 15 - 163 - 81|

SBP* denotes the results from my implementation, I believe the results can be improved by hyperparameter tuning.
