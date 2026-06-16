# ML_assn_04: Facial Expression Recognition

## პროექტის მიმოხილვა

ეს პროექტი არის Kaggle-ის **Challenges in Representation Learning: Facial Expression Recognition Challenge**-ზე.
დავალების მიზანია 48×48 შავ-თეთრი სახის გამოსახულებების კლასიფიკაცია 7 ემოციად:

| ინდექსი | ემოცია |
|---|---|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

გამოვიყენეთ FER2013 dataset — 28,709 train და 7,178 test სურათი. კლასების განაწილება არათანაბარია:
**Happy** (7,215) და **Neutral** (4,965) დომინირებენ, ხოლო **Disgust** მხოლოდ 436 ჩანაწერით არი (1.5%) წარმოდგენილი.
ეს რთულს ხდის ყველა კლასის თანაბრად კარგად სწავლას, განსაკუთრებით პატარა CNN-ებისთვის.

### მიდგომა

მოდელების აგება დავიწყეთ მარტივი **TinyCNN**-ით და ეტაპობრივად გავზარდეთ სიღრმე და სირთულე — **MediumCNN** -> **DeepCNN** -> **ResNet18 Transfer Learning**.
თითოეულ არქიტექტურაზე ცალ-ცალკე გავუშვით ჰიპერპარამეტრების ექსპერიმენტები (learning rate, optimizer, batch size, dropout, scheduler).
ყველა run დალოგილია **Weights & Biases**-ში `group`-ების მიხედვით.

პირველ ეტაპზე forward/backward pass-ებიც შევამოწმეთ — დავრწმუნდით, რომ gradient-ები ეცემა ყველა ფენაზე და optimizer რეალურად ანახლებს წონებს.

---

## რეპოზიტორიის სტრუქტურა

| ფაილი | აღწერა |
|---|---|
| `model_experiment.ipynb` | მონაცემების მომზადება, CNN/ResNet არქიტექტურები, ჰიპერპარამეტრების ექსპერიმენტები, forward/backward შემოწმება |
| `model_inference.ipynb` | საუკეთესო checkpoint-ის ჩატვირთვა, test set-ზე პროგნოზი და Kaggle submission-ის გენერაცია |
| `README.md` | პროექტის სრული აღწერა |

---

## Dataset და Preprocessing
**სვეტები:**
- `emotion` — target label (0–6)
- `pixels` — 2,304 space-separated pixel value (48x48)

**Preprocessing:**
- pixel string -> NumPy array -> reshape `(48, 48)`
- ნორმალიზაცია `[0, 1]` დიაპაზონში (`/ 255`)
- PyTorch tensor `(1, 48, 48)` — ერთი grayscale channel
- train/validation split: **80/20**, `stratify=emotion`, `random_state=42`
  - Train: 22,967 | Validation: 5,742
- Loss: `CrossEntropyLoss` (7-კლასიანი კლასიფიკაცია)

**DataLoader:**
- default batch size: 64 (ექსპერიმენტებში 32/128-იც გამოვიყენეთ)
- train: `shuffle=True`, val/test: `shuffle=False`

---

## Forward / Backward შემოწმება

TinyCNN-ზე (და შემდეგ ResNet18-ზეც) ლექციაში განხილული სტრატეგიით შევამოწმეთ, რომ მოდელი სწორად მუშაობს:

**Forward check:**
- Input: `(64, 1, 48, 48)` -> Output: `(64, 7)`
- საწყისი loss = `-log(1/7) = 1.946` (random guess 7 კლასზე)

**Backward check:**
- `loss.backward()`-ის შემდეგ ყველა conv/fc ფენაზე gradient არის არანულოვანი
- მაგ: `fc_layers.4.bias: grad_mean=0.052706`

**Optimizer check:**
- `optimizer.step()`-ის შემდეგ წონები იცვლება (`Weights changed: True`)

**Overfit test (1 batch, 50 epoch):**
- 64 სურათზე TinyCNN 50 epoch-ში 95.31% accuracy-მდე მივიდა
---

## CNN არქიტექტურები (ეტაპობრივი აგება)

### 1. TinyCNN (Baseline)

პირველი, შედარებით პატარა CNN — საწყისი baseline.

```
Conv(1->32) -> ReLU -> MaxPool
Conv(32->64) -> ReLU -> MaxPool
Flatten -> Linear(9216->128) -> ReLU -> Dropout(0.5) -> Linear(128->7)
```

**რატომ ასე:** მინიმალური არქიტექტურა, რომელიც სწრაფად ტრეინდება და საშუალებას გვაძლევს დავინახოთ, 
რამდენად საკმარისი არის მარტივი CNN ამ დატასეტისთვის.

| მეტრიკა | მნიშვნელობა |
|---|---|
| Best Val Acc | **52.87%** (epoch 19) |
| Train Acc (epoch 20) | 65.36% |
| Gap | 12.5% — ზომიერი underfitting |

**ანალიზი:** მოდელი სწავლობს, მაგრამ capacity შეზღუდულია — 
train acc 65%-ზე ადის, val acc კი მხოლოდ 53%-მდე. რთული ემოციები (Disgust, Fear) და
noisy 48×48 სურათები მარტივი 2-layer CNN-ით ვერ ისწავლება.

---

### 2. MediumCNN

Baseline-ის გაფართოება — BatchNorm, მესამე conv block, ორი FC ფენა.

```
Conv(1->32) -> BN -> ReLU -> MaxPool
Conv(32->64) -> BN -> ReLU -> MaxPool
Conv(64->128) -> BN -> ReLU -> MaxPool
Flatten -> Linear(4608->256) -> ReLU -> Dropout(0.5)
       -> Linear(256->64) -> ReLU -> Dropout(0.3) -> Linear(64->7)
```

**რატომ ასე:** BatchNorm სწავლებას ასტაბილურებს; დამატებითი conv block უფრო 
აბსტრაქტულ ფიჩერებს იჭერს; dropout overfitting-ს აკონტროლებს.

**Baseline run (Adam, lr=0.001):** Best Val Acc **55.85%** (epoch 15)

---

### 3. DeepCNN

ორმაგი conv ფენები თითო block-ში + 256-channel ბოლო block.

```
[Conv->BN->ReLU]×2 -> MaxPool  (32 channels)
[Conv->BN->ReLU]×2 -> MaxPool  (64 channels)
[Conv->BN->ReLU]×2 -> MaxPool  (128 channels)
Conv->BN->ReLU -> MaxPool      (256 channels)
Flatten -> FC(2304->256) -> Dropout -> FC(256->64) -> Dropout -> FC(64->7)
```

**რატომ ასე:** უფრო ღრმა receptive field და მეტი capacity — დატასეტის რთული facial ფიჩერებისთვის. 
თუმცა ბევრი პარამეტრი overfitting-ის რისკსაც ზრდის.

| Run | Best Val Acc | Train Acc (ep.20) | Gap | სტატუსი                     |
|---|---|---|---|-----------------------------|
| `deep-cnn-baseline` | **59.61%** | 87.86% | 28% | overfitting                 |
| `deep-cnn-higher-dropout` (0.6/0.5) | **58.99%** | 78.16% | 19% | უკეთესი ბალანსი             |
| `deep-cnn-even-higher-dropout` (0.7/0.5) | 55.02% | 72.58% | 17% | underfitting (too much reg) |
| `deep-cnn-cosine-lr` | **58.95%** | 86.08% | ~27% | საუკეთესო tuned run         |

---

### 4. ResNet18 Transfer Learning

ImageNet-ზე წინასწარ გაწვრთნილი ResNet18, grayscale-ისთვის ადაპტირებული:
- `conv1`: 3-channel -> 1-channel (pretrained weights-ის channel-wise საშუალო)
- `fc`: `Dropout(0.5) -> Linear(512→7)`

**ორი ეტაპი:**

1. **Fixed features** — backbone frozen, მხოლოდ FC ტრეინდება (10 epoch, lr=0.01)
2. **Fine-tuning** — ყველა ფენა unfrozen (20 epoch, lr=0.001, CosineAnnealingLR)

| Run | Best Val Acc | Train Acc (ep.20) | Gap | სტატუსი |
|---|---|---|---|---|
| `resnet18-fixed-features` | **27.17%** | 25% | 0% | underfitting |
| `resnet18-finetune` | **59.93%** | 97.67% | 38% | overfitting |

**ანალიზი:** frozen backbone-ით მოდელი ვერ სწავლობს — ImageNet feature-ები 48×48 grayscale ემოციებზზე 
პირდაპირ არ გადადის (underfitting). fine-tuning-ით val acc იზრდება, მაგრამ train acc 97.67%-მდე — 
კლასიკური overfit. val loss epoch 12-ის შემდეგ იზრდება (1.15 -> 2.42).

---

## ჰიპერპარამეტრების ექსპერიმენტები (MediumCNN)

MediumCNN გამოვიყენეთ საწვრთნელ არქიტექტურად — სწრაფად ტრეინდება და ჰიპერპარამეტრების ეფექტი მკაფიოდ ჩანს.

### Learning Rate

| LR | Best Val Acc (ep.20) | Train Acc | ანალიზი |
|---|---|---|---|
| 0.0001 | 54.23% | 71.64% | ნელი სწავლა, underfitting |
| **0.001** | **55.85%** | 66.28% | optimal |
| 0.01 | 25.13% | 25.13% | diverged — ვერ სწავლობს |
| 0.1 | 25.13% | 25.13% | diverged — ძალიან დიდი LR |

### Optimizer (lr=0.001, 20 epoch)

| Optimizer | Val Acc (ep.20) | Train Acc |
|---|---|---|
| SGD | 44.36% | 41.97% |
| SGD + Momentum | 53.29% | 63.73% |
| **NAG** | **56.51%** | 65.82% |
| Adagrad | 52.12% | 52.46% |
| Adadelta | 36.03% | 32.40% |
| RMSprop | 55.92% | 61.68% |

NAG (Nesterov) ყველაზე კარგი შედეგი მოგვცა — შემდეგი DeepCNN ექსპერიმენტებისთვისაც ის გამოვიყენეთ.

### Batch Size (NAG, lr=0.001)

| Batch Size | Val Acc (ep.20) | Train Acc |
|---|---|---|
| **32** | **57.12%** | 68.43% |
| 64 | 56.20% | 64.90% |
| 128 | 54.39% | 59.31% |

პატარა batch size უფრო noisy gradient-ს იძლევა, რაც ამ dataset-ზე რეგულარიზაციის ეფექტს აქვს.

### Regularization (batch=32)

| სტრატეგია | Val Acc | Train Acc | Gap | ანალიზი |
|---|---|---|---|---|
| no-dropout | 53.97% | **93.39%** | ~39% | მძიმე overfitting |
| current (0.5/0.3) | **57.33%** | 68.15% | 11% | optimal |
| high (0.6/0.4) | 55.12% | 61.49% | 6% | ცოტა underfitting |
| weight-decay (1e-4) | 56.65% | 68.63% | 12% | dropout-თან comparable |

**no-dropout** run-ი საუკეთესო overfitting მაგალითია: train 93%, val 54% — მოდელი training data-ს ოზეპირებს, 
validation-ზე კი ვერ განზოგადდდება.

---

## DeepCNN — საბოლოო Tuning

MediumCNN-ის ექსპერიმენტებიდან გამომდინარე, DeepCNN-ის საუკეთესო კონფიგურაცია:

| პარამეტრი | მნიშვნელობა |
|---|---|
| Optimizer | SGD + Nesterov (momentum=0.9) |
| Learning rate | 0.001 |
| LR Scheduler | CosineAnnealingLR (T_max=20) |
| Batch size | 32 |
| Dropout | 0.6 / 0.5 |
| Weight decay | 1e-4 |
| Epochs | 20 |

**Run:** `deep-cnn-cosine-lr` — Best Val Acc: **58.95%** (epoch 17)

Cosine scheduler LR-ს ეპოქების მსვლელობასთან ერთად ამცირებს, რაც fine-tuning ფაზაში convergence-ს უწყობს ხელს.

---

## მოდელების შედარება

| მოდელი | Best Val Acc | Train Acc (best/final) | Gap 
|---|---|---|---|
| TinyCNN | 52.87% | 65.36% | 12% 
| MediumCNN (tuned) | 57.33% | 68.15% | 11% 
| DeepCNN (baseline) | 59.61% | 87.86% | 28% 
| **DeepCNN (cosine-lr)** | **58.95%** | 86.08% | 27% 
| ResNet18 (fixed) | 27.17% | 25% | 0% |
| ResNet18 (finetune) | **59.93%** | 97.67% | 38% 

**საუკეთესო Val Accuracy:** ResNet18 fine-tune (59.93%)

**Submission-ისთვის არჩეული:** DeepCNN + cosine scheduler (58.95%)

ResNet18-მა validation-ზე უკეთესი შედეგი მოგვცა, მაგრამ train acc 97.67% და val loss-ის ზრდა epoch 12-ის შემდეგ 
მიუთითებს overfitting-ზე. DeepCNN-ის gap ნაკლებია და checkpoint-ი (`best_deep-cnn-cosine-lr.pt`) WandB artifact-ადაც 
ატვირთულია — ამიტომ submission-ისთვის ის ავირჩიეთ.

---

## Weights & Biases Tracking

ყველა ექსპერიმენტი: **https://wandb.ai/ml_assn_04/ml_assn_04**

**სტრუქტურა (group = არქიტექტურა):**

| Group | Run-ები |
|---|---|
| `TinyCNN` | `exp01-tiny-cnn` |
| `MediumCNN` | `exp02-medium-cnn-lr0.001`, `medium-cnn-lr*`, `medium-cnn-opt-*`, `medium-cnn-batch_size*`, `medium-cnn-reg-*` |
| `DeepCNN` | `deep-cnn-baseline`, `deep-cnn-higher-dropout`, `deep-cnn-even-higher-dropout`, `deep-cnn-cosine-lr` |
| `ResNet18Transfer` | `resnet18-fixed-features`, `resnet18-finetune` |

**თითოეული run-ისთვის დალოგილი:**
- `train_loss`, `val_loss`, `train_acc`, `val_acc`
- `best_val_acc`, `final_best_epoch` (scheduler version-ში)
- `lr` (CosineAnnealingLR run-ებში)
- hyperparameters (`config`: architecture, epochs, lr, batch_size, optimizer, dropout, weight_decay)
- best checkpoint -> `best_{run_name}.pt`

საუკეთესო run-ზე (`deep-cnn-cosine-lr`) validation set-ზე `model_inference.ipynb`-ში run-ის resume-ით WandB-ზე ასევე 
დავლოგე **confusion matrix** და **predictions table** (`actual`, `predicted_label`, `predicted_idx`).

---

## Kaggle Submission

`model_inference.ipynb`:
1. ტვირთავს `best_deep-cnn-cosine-lr.pt` checkpoint-ს
2. DeepCNN (`dropout1=0.6, dropout2=0.5`) test set-ზე inference
3. ქმნის `submission.csv` (7,178 prediction)
4. WandB-ზე artifact-ად ატვირთავს (`FER_submission`)


---
