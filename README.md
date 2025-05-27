<div align="center">
  <div>
    <h1>
        TSC:Efficient FRB Signal Search by a Two-stage Cascade Deep Learning Model on FAST
    </h1>
  </div>
  <div>
      <a href='https://github.com/lingshijiang'>Ronghuan Yan</a> &emsp; 
      <a href='https://github.com/aoxipo'>Junlin Li</a> &emsp; 
      <a href='https://it.ctgu.edu.cn/info/1551/50381.htm'>Weixin Tian*</a> &emsp;
      <a href='https://indussky8.github.io/'>Nyasha Mkwanda</a>
  </div>
  <br/>
</div>

TSC is frb classfiy deeplearning method based on the paper "TSC: Efficient FRB Signal Search by a Two-stage Cascade Deep Learning Model on FAST", which has been accepted by RAA 2025.

## Framework

![Overview](.\log\FrameWork.png)

TSC aims to address the challenge of rapid search for Fast Radio Bursts (FRBs) in large-scale observational data, where the scarcity of FRB signals and the vast volume of data hinder efficient detection.   To this end, TSC employs a two-stage deep learning framework optimized for FRB signal characteristics.   Inspired by the structural isomorphism of FRB waveforms, TSC first performs a coarse-grained screening to identify candidate signals quickly, followed by a fine-grained analysis to refine detection accuracy and reduce false positives, enhancing interpretability within the FRB Search framework.

# Environment

```
GPUtil
numpy==1.21.5
torch==1.12.0
opencv-python
astropy #using conda install astropy
h5py
```

# Dataset

We have produced simulation datasets of different specifications and labeled real datasets, please send email to 407157175@qq.com

# Train 

run python train.py in cmd with config

```
batch_size = 32
train_dir_path = "Path/To/DataSet"
data_shape = (4096, 4096)
method_dict ={
    "conv17":0,
    "inceptionresnetv2":1,
    "dense121":2,
    "efficientnet":3,
    "cmt":4,
    "ADFP":5,
    "PCT":6,
}

trainer = Train(
    image_shape = data_shape,
    class_number = 2, 
    is_show = False,
    name = "PCT",
    method_type = 6
)
```

run python ./TFR/model/torch_linear/train.py in cmd for GA model and DDPM Score Match



# Predict

run run_predict.py  with parameters

```python
'--model', default=2, choices=[0, 1, 2, 3, 4]
 ## Name of model to predict
 ## 0 -- conv17
 ## 1 -- inceptionresnetv2
 ## 2 -- dense121
 ## 3 -- efficientnet
 ## 4 -- cmt')
 
'--save_path', default='./predict_ans/'
 ## 'predict data save path'

'--data_dir', default="/home/data/lijl/DATA/Frbdata/fast/", 
## 'data dir path'

'--need_code', default=False, type=bool, 
## 'save map code and save print data'
```

# Log

Please refer to log folder for our experimental parameters and log results

# Effectiveness On Real DataSet

The data is based on publicly available samples from FAST

| <img src=".\log\hotmap\4_0.jpg" alt="4_0" width="245" height="245" /> | <img src=".\log\hotmap\4_0_mask.jpg" alt="4_0_mask" width="245" height="245" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src=".\log\hotmap\11.jpg" alt="11" width="245" height="245" /> | <img src=".\log\hotmap\11_mask.jpg" alt="11_mask" width="245" height="245" /> |
| <img src=".\log\hotmap\7_0.jpg" alt="7_0" width="245" height="245"/> | <img src=".\log\hotmap\7_0_mask.jpg" alt="7_0_mask" width="245" height="245" /> |
| <img src=".\log\hotmap\28_0.jpg" alt="28_0" width="245" height="245"/> | <img src=".\log\hotmap\28_0_mask.jpg" alt="28_0" width="245" height="245" /> |

## Citation

Please cite this work if you find it useful:

```latex
@article{Yan_2025,
doi = {10.1088/1674-4527/adb36c},
url = {https://dx.doi.org/10.1088/1674-4527/adb36c},
year = {2025},
month = {mar},
publisher = {National Astromonical Observatories, CAS and IOP Publishing},
volume = {25},
number = {3},
pages = {035007},
author = {Yan, Ronghuan and Li, Junlin and Tian, Weixin and Mkwanda, Nyasha},
title = {TSC: Efficient FRB Signal Search by a Two-stage Cascade Deep Learning Model on FAST},
journal = {Research in Astronomy and Astrophysics},
abstract = {Fast Radio Bursts (FRBs) have emerged as one of the most intriguing and enigmatic phenomena in the field of radio astronomy. The key of current related research is to obtain enough FRB signals. Computer-aided search is necessary for that task. Considering the scarcity of FRB signals and massive observation data, the main challenge is about searching speed, accuracy and recall. in this paper, we propose a new FRB search method based on Commensal Radio Astronomy FAST Survey (CRAFTS) data. The CRAFTS drift survey data provide extensive sky coverage and high sensitivity, which significantly enhance the probability of detecting transient signals like FRBs. The search process is separated into two stages on the knowledge of the FRB signal with the structural isomorphism, while a different deep learning model is adopted in each stage. To evaluate the proposed method, FRB signal data sets based on FAST observation data are developed combining simulation FRB signals and real FRB signals. Compared with the benchmark method, the proposed method F-score achieved 0.951, and the associated recall achieved 0.936. The method has been applied to search for FRB signals in raw FAST data. The code and data sets used in the paper are available at github.com/aoxipo.}
}

```

## Contact

If you have any specific questions or if there's anything else you'd like assistance with regarding the code, feel free to let me know. 
