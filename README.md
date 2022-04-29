## GCN-FFNN
Implementation code of the paper: "GCN-FFNN: A Two-Stream Deep Model for Learning Solution to Partial Differential Equations" [(ArXiv link)](https://arxiv.org/abs/2106.14742).


## Methodology
![pde-gcn](images/methodology.png)
<br />

## GCN Architecture
![pde-gcn <](images/architecture.png)
<br />

## Usage
Install the required packages with `pip install -r requirements.txt`.

Navigate to the desired folder, e.g. `pde-gcn/1d-burgers/ensemble-outer/`.

For training run, e.g.:
```
python ensemble-inner.py
```
For testing run, e.g.:
```
python ensemble-inner.py --test
```


<!-- 
## Citation 
```
```
-->