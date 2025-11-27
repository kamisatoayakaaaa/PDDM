# Experiments Log

## 2025-11-26 10:21:39 · dmih_teacher_coco128

- Working dir: `C:\Users\17007\Desktop\PDDM\PDDm`
- Command: `python DMIH/train_teacher_coco128.py --data_root .\DMIH\data\train2017 --image_size 256 --batch_size 8 --epochs 10`
- Git branch: `main`
- Git commit: `51572cf9aa410fcfc8c4332ee0a72f5ee07fd917`

### Config
- data_root: .\DMIH\data\train2017
- image_size: 256
- batch_size: 8
- epochs: 10
- timesteps: 1000
- lr: 0.0001
- save_dir: checkpoints/dmih_teacher_coco128

### Metrics
- best_train_loss: 0.08023958932608366
- final_train_loss: 0.08023958932608366

---

## 2025-11-26 10:32:18 · dmih_teacher_coco128

- Working dir: `C:\Users\17007\Desktop\PDDM\PDDm`
- Command: `python DMIH/train_teacher_coco128.py --data_root .\DMIH\data\train2017 --image_size 256 --batch_size 8 --epochs 5`
- Git branch: `main`
- Git commit: `51572cf9aa410fcfc8c4332ee0a72f5ee07fd917`

### Config
- data_root: .\DMIH\data\train2017
- image_size: 256
- batch_size: 8
- epochs: 5
- timesteps: 1000
- lr: 0.0001
- save_dir: checkpoints/dmih_teacher_coco128

### Metrics
- best_train_loss: 0.104759372305125
- final_train_loss: 0.104759372305125

---

## 2025-11-26 10:39:12 · dmih_teacher_coco128_eval

- Working dir: `C:\Users\17007\Desktop\PDDM\PDDm`
- Command: `python DMIH/evaluatediff.py --ckpt checkpoints/dmih_teacher_coco128/teacher_best.pt --data_root .\DMIH\data\train2017 --out_dir eval/teacher_best --num_pairs 8`
- Git branch: `main`
- Git commit: `51572cf9aa410fcfc8c4332ee0a72f5ee07fd917`

### Config
- ckpt: checkpoints/dmih_teacher_coco128/teacher_best.pt
- data_root: .\DMIH\data\train2017
- out_dir: eval\teacher_best
- num_pairs: 8
- image_size: 256
- timesteps: 1000
- seed: 0

---

## 2025-11-26 11:16:06 · dmih_teacher_coco128

- Working dir: `C:\Users\17007\Desktop\PDDM\PDDm`
- Command: `python DMIH/train_teacher_coco128.py --data_root .\DMIH\data\train2017 --image_size 256 --batch_size 8 --epochs 20`
- Git branch: `main`
- Git commit: `51572cf9aa410fcfc8c4332ee0a72f5ee07fd917`

### Config
- data_root: .\DMIH\data\train2017
- image_size: 256
- batch_size: 8
- epochs: 20
- timesteps: 1000
- lr: 0.0001
- exp_name: diffteachertrain_01
- exp_dir: C:\Users\17007\Desktop\PDDM\PDDM\DMIH\experiments\diffteachertrain_01
- ckpt_dir: C:\Users\17007\Desktop\PDDM\PDDM\DMIH\experiments\diffteachertrain_01\checkpoints

### Metrics
- best_train_loss: 0.04726231598760933
- final_train_loss: 0.05408609681762755

---

## 2025-11-26 13:35:43 · dmih_teacher_coco128_eval

- Working dir: `C:\Users\17007\Desktop\PDDM\PDDm`
- Command: `python DMIH/evaluatediff.py --ckpt DMIH\experiments\diffteachertrain_01\checkpoints\best.pt --data_root .\DMIH\data\train2017 --out_dir eval/teacher_best --num_pairs 8`
- Git branch: `main`
- Git commit: `51572cf9aa410fcfc8c4332ee0a72f5ee07fd917`

### Config
- ckpt: DMIH\experiments\diffteachertrain_01\checkpoints\best.pt
- data_root: .\DMIH\data\train2017
- out_dir: eval\teacher_best
- num_pairs: 8
- image_size: 256
- timesteps: 1000
- seed: 0

---

## 2025-11-26 13:52:06 · dmih_teacher_coco128_eval

- Working dir: `C:\Users\17007\Desktop\PDDM\PDDm`
- Command: `python DMIH/evaluatediff.py --ckpt DMIH\experiments\diffteachertrain_01\checkpoints\best.pt --data_root .\DMIH\data\train2017 --out_dir eval/teacher_best --num_pairs 8`
- Git branch: `main`
- Git commit: `51572cf9aa410fcfc8c4332ee0a72f5ee07fd917`

### Config
- ckpt: DMIH\experiments\diffteachertrain_01\checkpoints\best.pt
- data_root: .\DMIH\data\train2017
- out_dir: eval\teacher_best
- num_pairs: 8
- image_size: 256
- timesteps: 1000
- seed: 0

---

## 2025-11-26 14:02:36 · dmih_teacher_coco128_eval

- Working dir: `C:\Users\17007\Desktop\PDDM\PDDm`
- Command: `python DMIH/evaluatediff.py --ckpt DMIH/experiments/diffteachertrain_01/checkpoints/best.pt --data_root DMIH/data/train2017 --out_dir DMIH/eval/diffteachertrain_01 --num_pairs 4`
- Git branch: `main`
- Git commit: `51572cf9aa410fcfc8c4332ee0a72f5ee07fd917`

### Config
- ckpt: DMIH/experiments/diffteachertrain_01/checkpoints/best.pt
- data_root: DMIH/data/train2017
- out_dir: DMIH\eval\diffteachertrain_01
- num_pairs: 4
- image_size: 256
- timesteps: 1000
- seed: 0

---

## 2025-11-26 14:10:37 · dmih_teacher_coco128

- Working dir: `C:\Users\17007\Desktop\PDDM\PDDm`
- Command: `python DMIH/train_teacher_coco128.py --data_root .\DMIH\data\train2017 --image_size 256 --batch_size 8 --epochs 3`
- Git branch: `main`
- Git commit: `51572cf9aa410fcfc8c4332ee0a72f5ee07fd917`

### Config
- data_root: .\DMIH\data\train2017
- image_size: 256
- batch_size: 8
- epochs: 3
- timesteps: 1000
- lr: 0.0001
- exp_name: diffteachertrain_01
- exp_dir: C:\Users\17007\Desktop\PDDM\PDDM\DMIH\experiments\diffteachertrain_01
- ckpt_dir: C:\Users\17007\Desktop\PDDM\PDDM\DMIH\experiments\diffteachertrain_01\checkpoints

### Metrics
- best_train_loss: 0.21695055719465017
- final_train_loss: 0.21695055719465017

---

