# Model

## Dependencies

- python==3.10.12
- torch==2.3.1
- numpy==1.24.3
- pandas==2.0.3

## Datasets

### Raw data:

ISP-Weibo : https://github.com/vonfeng/DPLink/tree/master

### Preprocessed data:

In the dataset folder, we preprocessed the ISP-Weibo. If the paper is accepted, we will publish all the datasets.

## Usage

The model running environment is python, and its usage is as follows:

```python
python main.py --gpu=0 --times=1 --a_dataset=isp --b_dataset=wb 
```

a_dataset: platform A

b_dataset: platform B