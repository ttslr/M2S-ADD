# Betray Oneself: A Novel Audio DeepFake Detection Model via Mono-to-Stereo Conversion
 

## Introduction
This is an implementation of the following paper.
> [Betray Oneself: A Novel Audio DeepFake Detection Model via Mono-to-Stereo Conversion.](https://www.isca-archive.org/interspeech_2023/liu23v_interspeech.html)
> InterSpeech'2023

 [Rui Liu](https://ttslr.github.io/), Jinhua Zhang, Guanglai Gao, [Haizhou Li](https://colips.org/~eleliha/).
 



## Installation
First, clone the repository locally, create and activate a conda environment, and install the requirements :
```
$ conda create --name M2S-ADD python=3.8.8
$ conda activate M2S-ADD
$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
$ pip install -r requirements.txt
```


## Experiments

### Dataset
Our experiments are done in the logical access (LA) partition of the ASVspoof 2019 dataset, which can can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3336).

### Training
To train the model run:
```
python main.py 
```

### Testing

To evaluate your own model on LA evaluation dataset:
```
python main.py --track=logical --loss=WCE --is_eval --eval --model_path='/path/to/your/best_model.pth' --eval_output='eval_CM_scores_file.txt'
```

If you would like to compute scores on development dataset simply run:
```
python main.py --track=logical --loss=WCE --eval --model_path='/path/to/your/best_model.pth' --eval_output='dev_CM_scores_file.txt'
```

Compute the min t-DCF and EER(%) on development dataset
```
python tDCF_python_v2/evaluate_tDCF_asvspoof19_eval_LA.py  dev  'dev_CM_scores_file.txt'
```

Compute the min t-DCF and EER(%) on evaluation dataset
```
python tDCF_python_v2/evaluate_tDCF_asvspoof19_eval_LA.py  Eval  'eval_CM_scores_file.txt'
```

## Acknowledgements

This repository is built on RawGAT-ST-antispoofing.
- https://github.com/eurecom-asp/RawGAT-ST-antispoofing

Authors would like to acknowledge other repositories as well.
- [min t-DCF implementation](https://www.asvspoof.org/resources/tDCF_python_v2.zip)
- [core scripts from Dr. Xin Wang, NII Japan](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts)



## Contact
For any query regarding this repository, please contact:
- Rui Liu (e-mail:liurui_imu@163.com), Jinhua Zhang(e-mail: zjh_imu@163.com)

## Citation
If you use M2S-ADD for anti-spoofing please use the following citations:
@article{Liu2023BetrayOA,
  title={Betray Oneself: A Novel Audio DeepFake Detection Model via Mono-to-Stereo Conversion},
  author={Rui Liu and Jinhua Zhang and Guanglai Gao and Haizhou Li},
  journal={ArXiv},
  year={2023},
  volume={abs/2305.16353},
  url={https://api.semanticscholar.org/CorpusID:258947124}
} 