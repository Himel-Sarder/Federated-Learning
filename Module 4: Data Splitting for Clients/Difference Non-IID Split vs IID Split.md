| Aspect                           | IID Split (Random)               | Non-IID Split (by Label/Class)             |
| -------------------------------- | -------------------------------- | ------------------------------------------ |
| Data distribution across clients | Similar                          | Different / skewed                         |
| Class balance per client         | Similar class ratios             | Skewed class ratios                        |
| Real-world realism               | Low                              | High                                       |
| FedAvg convergence               | Fast, stable                     | Slow, unstable                             |
| Final accuracy                   | Higher                           | Usually lower                              |
| Client drift                     | Low                              | High                                       |
| Training stability               | Stable                           | Fluctuates across rounds                   |
| Example                          | Each hospital has mixed patients | Each hospital specializes in certain cases |

IID split:
Each client looks like a small version of the full dataset.
FedAvg works well because client updates point in similar directions.

Non-IID split:
Each client sees a different data distribution.
FedAvg aggregates conflicting updates, causing:
- slower learning
- oscillations
- lower final performance
