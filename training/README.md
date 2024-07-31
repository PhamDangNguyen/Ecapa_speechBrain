
# Introduction

## Model pretrain được sử dụng trên VoxLingua107 dataset
Folder này có những scripts support chạy train một detect language model có tên là Ecapa của SpeechBrain với example là VoxLingua107 dataset.

Tại sao nên dùng: Model khá nhẹ infer nhanh => fine-tune với các task yêu cầu thời gian phản hồi nhanh khá tốt.

## Cài đặt môi trường cùng libraries
Chạy các câu lệnh sau để cài tất cả các Dependencies:
```python
pip install -r extra_requirements.txt
pip install -r lint-requirements.txt
pip install -r requirements.txt
pip install pillow
```

## Prepare data
### Downloading and making the data
- File ```make_data_Identify_huggingface.ipynb``` giúp người dùng lấy được data public từ Hugging face để fine-tune (đa phần là data English) để bỏ vào folder ``en``.
- File ```make_data_Identify.ipynb``` giúp coppy data Vietnamese đã có để bỏ vào folder ``vi``.

=>  Data hiện tại được cân bằng với số lượng file của mỗi folder en và vi khoảng 220 nghìn file.

### Convert data phục vụ training
Các file âm thanh phải tuân theo quy tắc đặt tên ```{vi=1 or en=0}.{duration}_{index}.wav```
#### Các thư mục data cần được bố trí như sau:
```
data/
├── vi/
|     └── 1_1.231234159_14.wav
|              . . . .
└── en/
       └── 0_3.343475201904_977.wav
              . . . . . .
```
Trong đó: đối với một đường dẫn ``<root>/data/vi/1_1.231234159.wav``
```
loc = <root>/data/vi/1_1.231234159.wav
lang = vi
key = vi/1_1_231234159
dur = 1.231234159
```
## Training

### Setup hparams file
Trong file .yaml cần setup lại các tham số sau:

#### 1. Output_folder: Nơi sinh ra file .log + train_log và lưu lại các version code.
#### 2. Save_folder: Nơi lưu file model .cpkt + file lable.
#### 3. Out_n_neurons: Với đơn vị neuron class cuối. Classification với n language thì ở đây Out_n_neurons = n
#### 4. Train_meta và val_meta: Đường dẫn tới file .json của train và val.
#### 5. Train_shards và val_shards: Đường dẫn tới file .tar của train và val.

### Train model
Run code:
```
python train.py hparams/train_ecapa.yaml
```

Training khoảng 100 epoch với batchsize khoảng 32.


# Performance



# Inference
The pre-trained model + easy inference is available on HuggingFace:
- https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa

You can run inference with only few lines of code:

```python
import torchaudio
from speechbrain.inference import EncoderClassifier
language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")
# Download Thai language sample from Omniglot and convert to suitable form
signal = language_id.load_audio("https://omniglot.com/soundfiles/udhr/udhr_th.mp3")
prediction =  language_id.classify_batch(signal)
print(prediction)
  (tensor([[-2.8646e+01, -3.0346e+01, -2.0748e+01, -2.9562e+01, -2.2187e+01,
         -3.2668e+01, -3.6677e+01, -3.3573e+01, -3.2545e+01, -2.4365e+01,
         -2.4688e+01, -3.1171e+01, -2.7743e+01, -2.9918e+01, -2.4770e+01,
         -3.2250e+01, -2.4727e+01, -2.6087e+01, -2.1870e+01, -3.2821e+01,
         -2.2128e+01, -2.2822e+01, -3.0888e+01, -3.3564e+01, -2.9906e+01,
         -2.2392e+01, -2.5573e+01, -2.6443e+01, -3.2429e+01, -3.2652e+01,
         -3.0030e+01, -2.4607e+01, -2.2967e+01, -2.4396e+01, -2.8578e+01,
         -2.5153e+01, -2.8475e+01, -2.6409e+01, -2.5230e+01, -2.7957e+01,
         -2.6298e+01, -2.3609e+01, -2.5863e+01, -2.8225e+01, -2.7225e+01,
         -3.0486e+01, -2.1185e+01, -2.7938e+01, -3.3155e+01, -1.9076e+01,
         -2.9181e+01, -2.2160e+01, -1.8352e+01, -2.5866e+01, -3.3636e+01,
         -4.2016e+00, -3.1581e+01, -3.1894e+01, -2.7834e+01, -2.5429e+01,
         -3.2235e+01, -3.2280e+01, -2.8786e+01, -2.3366e+01, -2.6047e+01,
         -2.2075e+01, -2.3770e+01, -2.2518e+01, -2.8101e+01, -2.5745e+01,
         -2.6441e+01, -2.9822e+01, -2.7109e+01, -3.0225e+01, -2.4566e+01,
         -2.9268e+01, -2.7651e+01, -3.4221e+01, -2.9026e+01, -2.6009e+01,
         -3.1968e+01, -3.1747e+01, -2.8156e+01, -2.9025e+01, -2.7756e+01,
         -2.8052e+01, -2.9341e+01, -2.8806e+01, -2.1636e+01, -2.3992e+01,
         -2.3794e+01, -3.3743e+01, -2.8332e+01, -2.7465e+01, -1.5085e-02,
         -2.9094e+01, -2.1444e+01, -2.9780e+01, -3.6046e+01, -3.7401e+01,
         -3.0888e+01, -3.3172e+01, -1.8931e+01, -2.2679e+01, -3.0225e+01,
         -2.4995e+01, -2.1028e+01]]), tensor([-0.0151]), tensor([94]), ['th'])
# The scores in the prediction[0] tensor can be interpreted as log-likelihoods that
# the given utterance belongs to the given language (i.e., the larger the better)
# The linear-scale likelihood can be retrieved using the following:
print(prediction[1].exp())
  tensor([0.9850])
# The identified language ISO code is given in prediction[3]
print(prediction[3])
  ['th']

# Alternatively, use the utterance embedding extractor:
emb =  language_id.encode_batch(signal)
print(emb.shape)
  torch.Size([1, 1, 256])
```


# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{ravanelli2024opensourceconversationalaispeechbrain,
      title={Open-Source Conversational AI with SpeechBrain 1.0},
      author={Mirco Ravanelli and Titouan Parcollet and Adel Moumen and Sylvain de Langen and Cem Subakan and Peter Plantinga and Yingzhi Wang and Pooneh Mousavi and Luca Della Libera and Artem Ploujnikov and Francesco Paissan and Davide Borra and Salah Zaiem and Zeyu Zhao and Shucong Zhang and Georgios Karakasidis and Sung-Lin Yeh and Pierre Champion and Aku Rouhe and Rudolf Braun and Florian Mai and Juan Zuluaga-Gomez and Seyed Mahed Mousavi and Andreas Nautsch and Xuechen Liu and Sangeet Sagar and Jarod Duret and Salima Mdhaffar and Gaelle Laperriere and Mickael Rouvier and Renato De Mori and Yannick Esteve},
      year={2024},
      eprint={2407.00463},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.00463},
}
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```

# **Citing VoxLingua107**
You can also cite the VoxLingua107 dataset paper if you use this model in research.

```bibtex
@inproceedings{valk2021slt,
  title={{VoxLingua107}: a Dataset for Spoken Language Recognition},
  author={J{\"o}rgen Valk and Tanel Alum{\"a}e},
  booktitle={Proc. IEEE SLT Workshop},
  year={2021},
}
```
# Ecapa_speechBrain
