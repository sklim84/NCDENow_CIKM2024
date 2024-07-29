# Bridging Dynamic Factor Models and Neural Controlled Differential Equations for Nowcasting GDP
This code is the official implementation of "Bridging Dynamic Factor Models and Neural Controlled Differential Equations for Nowcasting GDP".

## Requirements
Run the following to install requirements:
```setup
conda env create --file environment.yaml
```

## Datasets

If you need to download the datasets, you can find them at the following locations:

* **KOR GDP dataset**: available on [Economics Statistics System](https://ecos.bok.or.kr).
* **UK GDP dataset**: available from [Anesti et al. (2018)](https://www.bankofengland.co.uk/working-paper/2018/uncertain-kingdom-nowcasting-gdp-and-its-revisions).


## Proposed Model Usage
* Train and evaluate NCDENow through `main_ncdenow.sh`:
```sh
\NCDENow\main_ncdenow.sh
```

## Baseline Models Usage
#### The variants of RNNs
* Train and evaluate NCDENow through `main_rnns.sh`:
```sh
\NCDENow\baselines\RNNs\main_rnns.sh
```

#### Dynamic Factor Model
* Train and evaluate NCDENow through `main_dfm.sh`:
```sh
\NCDENow\baselines\DFM\main_dfm.sh
```

## Additional Information
* You can review the evaluation results at the following path. It is saved in the CSV file format. 
```bash
\NCDENow\_results\cikm2024\{DATASET_NAME}_{MODEL_NAME}_metrics.csv
```
* You can review the logs related to the training of the model at the following path:
```bash
\NCDENow\_logs\cikm2024\{MODEL_NAME}_{YYYYMMDD}.log
```
