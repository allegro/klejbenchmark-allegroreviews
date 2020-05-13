# Allegro Reviews
**Allegro Reviews** is a sentiment analysis dataset, consisting of 11,588 product reviews written in Polish and extracted from [Allegro.pl](https://allegro.pl) - a popular e-commerce marketplace. Each review contains at least 50 words and has a rating on a scale from one (negative review) to five (positive review).

We recommend using the provided train/dev/test split. The ratings for the test set reviews are kept hidden. You can evaluate your model using the online evaluation tool available on [klejbenchmark.com](https://klejbenchmark.com/).

The dataset can be downloaded from [here](https://klejbenchmark.com/static/data/klej_ar.zip).

## Evaluation
To counter slight class imbalance in the dataset, we propose to evaluate models using wMAE, i.e.macro-average of the mean absolute error per class. Additionally, we transform the rating to be between zero and one and report 1 − wMAE to obtain the final score.

Python implementation of the proposed metric:
```python
import pandas as pd
from sklearn.metrics import mean_absolute_error


def ar_score(y_true, y_pred):
    ds = pd.DataFrame({
        'y_true': (y_true - 1.0)/4.0, 
        'y_pred': (y_pred - 1.0)/4.0,
    })
    wmae = ds \
        .groupby('y_true') \
        .apply(lambda df: mean_absolute_error(df['y_true'], df['y_pred'])) \
        .mean()

    return 1 - wmae
```

## Results
| Model             | AR Score  |
| ----------------- | --------: |
| [ELMo](https://clarin-pl.eu/dspace/handle/11321/690) | **86.15** |
| [Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md) | 83.33 |
| [Slavic BERT](https://github.com/deepmipt/Slavic-BERT-NER) | 84.31   |
| [XLM-17](https://github.com/facebookresearch/XLM/#the-17-and-100-languages) | 84.52 |
| [HerBERT](https://github.com/allegro/HerBERT) | 84.48 |

## License
CC BY-SA 4.0

## Citation
If you use this dataset, please cite the following paper:
```
@misc{rybak2020klej,
    title={KLEJ: Comprehensive Benchmark for Polish Language Understanding},
    author={Piotr Rybak and Robert Mroczkowski and Janusz Tracz and Ireneusz Gawlik},
    year={2020},
    eprint={2005.00630},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
Paper is accepted at ACL 2020. We will update the BibTeX as soon as proceedings appear.

## Authors
Dataset was created by the **Allegro Machine Learning Research** team.

You can contact us at: <a href="mailto:klejbenchmark@allegro.pl">klejbenchmark@allegro.pl</a>
