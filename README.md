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

# Citation
<pre>
  @inproceedings{yan2025correlation,
  title={Correlation-Attention Masked Temporal Transformer for User Identity Linkage Using Heterogeneous Mobility Data},
  author={Yan, Ziang and Zhao, Xingyu and Ma, Hanqing and Chen, Wei and Qi, Jianpeng and Yu, Yanwei and Dong, Junyu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={12},
  pages={12999--13007},
  year={2025}
}
</pre>
