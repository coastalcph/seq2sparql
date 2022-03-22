# Compositional Wikidata Questions (CWQ)
CWQ is a multilingual KBQA dataset grounded in and executable over
Wikidata. Our dataset includes questions in four languages (Hebrew, Kannada, Chinese and English), and their associated SPARQL
queries.

## Data
The first version of CWQ dataset accompanying the preprint [Multilingual Compositional Wikidata Questions](https://arxiv.org/pdf/2108.03509.pdf) is stored in the repo as `cwq/dataset.json`. 

The three MCD splits and a random split is stored under `cwq/split/`.

The dataset's details and generation method is described in the arXiv preprint.

---
We are planning to release a second version of CWQ, named MCWQ. We create the dataset in a similar approach as CWQ, but with a finer wikidata property mapping  and data cleaning strategy. MCWQ contains 124,187 question query pairs, which is about 11 times the size of CWQ. 

You can download the latest dataset and its splits on [Google Drive](https://drive.google.com/drive/folders/19YmuXYKmnmVllVkOr9nMT1nsXFd9i9hu?usp=sharing).  

Below show the results of monolingual and cross-lingual experiments on MCWQ. 

## Experiment Results
**Monolingual Experiments**:

|         **Exact Match (%)**                     |  MCD1  |       |       |       | MCD2   |       |       |       | MCD3   |       |       |       | MCD_mean   |       |       |       | Random |       |       |       |
| :---------------------------- | :------------: | :---: | :---: | :---: | :------------: | :---: | :---: | :---: | :------------: | :---: | :---: | :---: | :---------------: | :---: | :---: | :---: | :----: | :---: | :---: | :---: |
|      **Language**           | En             | He    | Kn    | Zh    | En             | He    | Kn    | Zh    | En             | He    | Kn    | Zh    | En                | He    | Kn    | Zh    | En     | He    | Kn    | Zh    |
| LSTM+Attention                | 38\.2          | 29\.3 | 27\.1 | 26\.1 | 6\.3           | 5\.6  | 9\.9  | 7\.5  | 13\.6          | 11\.5 | 15\.7 | 15\.1 | 19\.4             | 15\.5 | 17\.6 | 16\.2 | 96\.6  | 80\.8 | 88\.7 | 86\.8 |
| E. Transformer                | 53\.3          | 35    | 30\.7 | 31    | 16\.5          | 8\.7  | 11\.9 | 10\.2 | 18\.2          | 13    | 18\.1 | 15\.5 | 29\.3             | 18\.9 | 20\.2 | 18\.9 | 99     | 90\.4 | 93\.7 | 92\.2 |
| mBERT/BERT                    | 49\.5          | 38\.7 | 34\.4 | 35\.6 | 13\.4          | 11\.4 | 12\.3 | 15\.1 | 17             | 18    | 18\.1 | 19\.4 | 26\.6             | 22\.7 | 21\.6 | 23\.4 | 98\.7  | **91**    | **95\.1** | **93\.3** |
| T5-base                       | 57\.4          | -     | -     | -     | 14\.6          | -     | -     | -     | 12\.3          | -     | -     | -     | 28\.1             | -     | -     | -     | 98\.5  | -     | -     | -     |
| mt5-small                     | **77\.6**          | 57\.8 | **55**    | **52\.8** | 13             | 12\.6 | 8\.2  | 21\.1 | **24\.3**          | 17\.5 | **31\.4** | 34\.9 | **38\.3**             | 29\.3 | 31\.5 | **36\.3** | 98\.6  | 90    | 93\.8 | 91\.8 |
| mT5-base                      | 55\.5          | **59\.5** | 49\.1 | 30\.2 | **27\.7**          | **16\.6** | **16\.6** | **23**    | 18\.2          | **23\.4** | 30\.5 | **35\.6** | 33\.8             | **33\.2** | **32\.1** | 29\.6 | **99\.1**  | 90\.6 | 94\.2 | 92\.2 |

**Zero-shot cross-lingual transfer**:

|      **Exact Match (%)**                         | MCD1   |       |       | MCD2   |       |       | MCD3   |       |       | MCD_mean   |       |       | Random |       |       |
| :---------------------------- | :------------: | :---: | :---: | :------------: | :---: | :---: | :------------: | :---: | :---: | :---------------: | :---: | :---: | :----: | :---: | :---: |
|             **Language**                              | He             | Kn   | Zh   | He             | Kn   | Zh   | He             | Kn   | Zh   | He                | Kn   | Zh   | He     | Kn   | Zh   |
| mT5-small                     | **0\.4**           | **0\.8**  | **0\.1**  | 0\.1           | 0\.1  | 0\.1  | **0\.1**           | **0\.1**  | 0\.1  | 0\.2              | 0\.3  | 0\.2  | 0\.5   | 0\.4  | 1\.1  |
| mT5-base                      | 0\.1           | 0     | 0     | **1\.0**           | **2\.2**  | **4\.1**  | **0\.1**           | 0     | **0\.3**  | **0\.4**              | **0\.7**  | **1\.5**  | **1\.1**   | **0\.9**  | **7\.2**  |
## Code for data generation and the experiments
Preparing in progress.

## Citations

If you use this dataset, please cite the following:
* Ruixiang Cui, Rahul Aralikatte, Heather Lent and Daniel Hershcovich.2021.[Multilingual Compositional Wikidata Questions](https://arxiv.org/pdf/2108.03509.pdf). _arXiv preprint
arXiv:2108.03509_.
``` bibtex
@misc{cui2021multilingual,
      title={Multilingual Compositional Wikidata Questions}, 
      author={Ruixiang Cui and Rahul Aralikatte and Heather Lent and Daniel Hershcovich},
      year={2021},
      eprint={2108.03509},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
The CWQ dataset is based on [CFQ](https://github.com/google-research/google-research/tree/master/cfq).

## Contact
For questions and usage issues, please contact <rc@di.ku.dk> .

## License
CWQ is released under the [CC-BY license](https://creativecommons.org/licenses/by/4.0/).