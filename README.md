# IEEE-CIS Fraud Detection — ML პროექტი

## პროექტის მიმოხილვა

ეს პროექტი წარმოადგენს Kaggle-ის კონკურსის [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) გადაწყვეტას. კონკურსის მიზანია ტრანზაქციების მონაცემებიდან გამოვავლინოთ თაღლითური ოპერაციები (Fraud). მონაცემები მოიცავს ტრანზაქციის ინფორმაციას (თანხა, ბარათის ტიპი, ელ-ფოსტის დომენი, V-ფიჩერები) და დამატებით Identity ინფორმაციას. შეფასების მეტრიკა არის **ROC-AUC**.

---

## ჩემი მიდგომა

პრობლემა გადავჭერი **gradient boosting** მოდელების ოჯახით (XGBoost, LightGBM, CatBoost), რომლებიც ტაბულური მონაცემებზე საუკეთესო შედეგს იძლევა. გატესტვა განვახორციელე Random Forest-ზე და Logistic Regression-ზე ბეისლაინებისთვის. ყველა ექსპერიმენტი დავალოგე **MLflow + DagsHub**-ზე.

---

## რეპოზიტორიის სტრუქტურა

```
├── model_experiment_XGBoost.ipynb          # XGBoost ექსპერიმენტი
├── model_experiment_LightGBM.ipynb         # LightGBM ექსპერიმენტი
├── model_experiment_CatBoost.ipynb         # CatBoost ექსპერიმენტი
├── model_experiment_RandomForest.ipynb     # Random Forest ექსპერიმენტი
├── model_experiment_LogisticRegression.ipynb # Logistic Regression ექსპერიმენტი
├── model_inference.ipynb                   # ინფერენსი + Kaggle submission
└── README.md                               # ეს ფაილი
```

### ფაილების განმარტება

| ფაილი | აღწერა |
|-------|--------|
| `model_experiment_XGBoost.ipynb` | XGBoost-ის სრული pipeline: cleaning → FE → FS → training (under/overfit + Optuna) → MLflow logging + Model Registry |
| `model_experiment_LightGBM.ipynb` | LightGBM-ის სრული pipeline, LightGBM-ის native categorical მხარდაჭერა |
| `model_experiment_CatBoost.ipynb` | CatBoost-ის pipeline, categorical ფიჩერების Pool-ით გადაცემა |
| `model_experiment_RandomForest.ipynb` | Random Forest ბეისლაინი, RF importance-ზე დაფუძნებული FS |
| `model_experiment_LogisticRegression.ipynb` | LR ბეისლაინი, SelectKBest + Lasso FS, StandardScaler |
| `model_inference.ipynb` | Model Registry-დან საუკეთესო მოდელის ჩამოტვირთვა, ტესტ სეტზე პრედიქცია, Kaggle submission-ის გენერაცია, ანსამბლი |

---

## Feature Engineering

### კატეგორიული ცვლადების დამუშავება

- **Label Encoding** — ყველა `object` ტიპის სვეტისთვის (XGBoost, LightGBM, RF, LR)
- **CatBoost Native** — CatBoost-ში categorical ფიჩერები Pool-ის მეშვეობით პირდაპირ გადაეცემა, Label Encoding სჭირდება
- **ელ-ფოსტის დომენი** — იშვიათი მნიშვნელობები (`P_emaildomain`, `R_emaildomain`) `'other'`-ში გაერთიანდა, Top-10 ან Top-15 ყველაზე ხშირი დომენი შენარჩუნდა

### NaN მნიშვნელობების დამუშავება

| მიდგომა | მოდელი | დასაბუთება |
|---------|--------|-----------|
| `-999` sentinel | XGBoost, LightGBM, RF | Gradient boosting-ი სწავლობს missing patterns-ს |
| `'missing'` string | CatBoost (კატეგ.) | CatBoost native NaN handling |
| Median imputation | Logistic Regression | LR-ს სჭირდება სრული მონაცემები, scale-sensitive |
| `nan_count` feature | ყველა | Missing-ის pattern-ი თავად ინფორმატიულია |

### შექმნილი ფიჩერები

**დროის ფიჩერები** (`TransactionDT` წამებშია):
- `hour` — დღის საათი (0-23)
- `day_of_week` — კვირის დღე (0-6)
- `is_night` — ღამის ტრანზაქცია (22:00-06:00) → **fraud-ის ინდიკატორი**
- `is_weekend` — შაბათ-კვირა → განსხვავებული ქცევის ნიმუში

**თანხის ფიჩერები**:
- `TransactionAmt_log` — log(1+x) ტრანსფორმაცია skewness-ის შემცირებისთვის
- `TransactionAmt_cents` — ცენტების ნაწილი (0.00 vs 0.99 შეიძლება fraud-ის ნიშანი)
- `TransactionAmt_isround` — მრგვალი თანხა (1 = round number)

**აგრეგაციული ფიჩერები** (card1, card2, addr1-ის მიხედვით):
- `{col}_amt_mean` — ბარათის/მისამართის საშუალო ტრანზაქციის თანხა
- `{col}_amt_std` — სტანდარტული გადახრა
- `{col}_amt_z` — Z-score (ანომალიური ტრანზაქცია ამ ბარათისთვის)

**სხვა**:
- `email_match` — P და R ელ-ფოსტის დომენების დამთხვევა (0/1)
- `card_combo` — card4 + card6 კომბინაცია
- `nan_count` — რიგში NaN-ების რაოდენობა

---

## Cleaning მიდგომები

### ჩამოწვეთილი სვეტები

| კრიტერიუმი | მოდელი | შედეგი |
|------------|--------|--------|
| >90% missing | XGBoost, LightGBM, CatBoost, RF | ~40-60 სვეტი ამოიღეს |
| >50% missing | Logistic Regression | უფრო მეტი სვეტი ამოიღო (LR-ს სჭირდება სუფთა მონაცემი) |
| Constant columns (nunique ≤ 1) | ყველა | 0 ვარიაციის სვეტები უსარგებლოა |
| Quasi-constant (>99.5% same) | LightGBM | თითქმის უცვლელი სვეტები |

### ელ-ფოსტის დომენი

იშვიათი დომენები (e.g. `yahoo.es`, `outlook.kr`) გაერთიანდა `'other'`-ში. ეს ამცირებს კარდინალობას და ხელს უშლის overfitting-ს იშვიათ კატეგორიებზე.

---

## Feature Selection

### გამოყენებული მიდგომები

#### 1. Model-Based Importance (XGBoost, LightGBM, CatBoost, RF)
- სწრაფი baseline მოდელი fitდება, feature importance გამოიყვანება
- Top 100-150 ფიჩერი ინარჩუნებს 95%+ variance-ს
- **შედეგი**: 400+ → 100-150 ფიჩერი, AUC თითქმის არ შეიცვლება

#### 2. Correlation Filter
- `corr > 0.95` მქონე ფიჩერებიდან ერთი ამოიღეს
- V-ფიჩერები (V1-V339) ძლიერ კორელირებულია ერთმანეთთან
- **შედეგი**: დამატებით 10-20 ფიჩერის ამოღება

#### 3. Zero Importance Removal (LightGBM)
- LightGBM-ი `importance=0` ანიჭებს სრულიად შეუსაბამო ფიჩერებს
- ეს ფიჩერები უბრალოდ ხმაურს ამატებს
- **შედეგი**: სუფთა ფიჩერ სეტი

#### 4. SelectKBest + L1 (Logistic Regression)
- `SelectKBest(f_classif, k=80)` — ANOVA F-score სტატისტიკა
- `L1 (Lasso, C=0.01)` — სასჯელი ნულოვდება irrelevant ფიჩერებს
- **Union** — ორივე მეთოდის გაერთიანება
- **შედეგი**: LR-ისთვის 80-120 ფიჩერი

#### შეფასება
| მეთოდი | Feature რაოდენობა | AUC (CV) | შენიშვნა |
|--------|------------------|----------|---------|
| ყველა ფიჩერი | 430+ | baseline | overfitting-ის რისკი |
| Top-150 importance | 150 | +0.001 | კარგი ბალანსი |
| + Correlation filter | 130 | +0.001 | ჭარბი ფიჩერები ამოღებულია |
| Zero importance removal | 120 | ≈ same | სისუფთავე |

---

## Training

### ტესტირებული მოდელები

| მოდელი | OOF AUC | შენიშვნა |
|--------|---------|---------|
| XGBoost (tuned) | ~0.926 | ყველაზე სწრაფი GPU-ზე |
| LightGBM (tuned) | ~0.928 | ოდნავ უკეთესი XGBoost-ზე |
| CatBoost (tuned) | ~0.924 | კარგია categorical ფიჩერებზე |
| Random Forest | ~0.890 | შედარებით სუსტი, მეხსიერება-ინტენსიური |
| Logistic Regression | ~0.830 | Baseline; მრავალი ფიჩერი არ-ხაზოვანია |
| Ensemble (weighted) | ~0.930 | საუკეთესო საერთო შედეგი |

### Underfitting vs Overfitting ანალიზი

#### Underfitted მოდელები (მაღალი Bias)

**XGBoost — max_depth=2, n_estimators=50**:
- CV AUC ~0.78 — მნიშვნელოვნად დაბლა optimum-ზე
- მიზეზი: ძალიან ზედაპირული ხეები ვერ სწავლობს ნელ ურთიერთქმედებებს
- ამ მონაცემებში fraud detection-ი საჭიროებს 6-8 სიღრმეს

**LightGBM — num_leaves=8, max_depth=3**:
- CV AUC ~0.79
- მიზეზი: ოდნავი გამყოფი ვერ ხსნის fraud pattern-ებს

**Logistic Regression — C=0.0001 (ძალიან ძლიერი Lasso)**:
- CV AUC ~0.80
- მიზეზი: regularization თრგუნავს ყველა კოეფიციენტს 0-სთან
- LR-ი ზოგადად underfits-ს ამ nonlinear მონაცემებზე

#### Overfitted მოდელები (მაღალი Variance)

**XGBoost — max_depth=12, no regularization**:
- Train AUC: 0.999 | CV AUC: 0.91 | **Gap: 0.09**
- მიზეზი: ძალიან ღრმა ხეები ისწავლება train-ის noise-ს
- reg_alpha=0, reg_lambda=0 — regularization-ის გარეშე

**LightGBM — num_leaves=500, min_child_samples=1**:
- Train AUC: 0.999 | CV AUC: 0.90 | **Gap: 0.09**
- მიზეზი: 500 ფოთოლი → ძალიან fine-grained splits → memorization

**Random Forest — max_depth=None, min_samples_leaf=1**:
- Train AUC: 1.0 | CV AUC: 0.88 | **Gap: 0.12**
- RF-ი სრულ სიღრმეზე ყოველთვის იმახსოვრებს train სეტს

### Hyperparameter ოპტიმიზაცია

**Optuna** — Bayesian Optimization (TPE Sampler):
- 20-30 trial თითო მოდელისთვის
- 5-Fold Stratified Cross Validation თითო trial-ზე
- ოპტიმიზებული პარამეტრები:

| პარამეტრი | ძიების სივრცე | საუკეთესო (LightGBM) |
|-----------|-------------|---------------------|
| n_estimators / iterations | 300-2000 | ~1200 |
| max_depth / depth | 4-12 | 7 |
| learning_rate | 0.01-0.2 | ~0.05 |
| num_leaves | 31-300 | ~127 |
| subsample | 0.5-1.0 | ~0.8 |
| reg_alpha | 1e-4 to 10 | ~0.5 |
| scale_pos_weight | 1-5 | ~2.5 (imbalanced data) |

**Cross Validation სტრატეგია**: 5-Fold Stratified KFold
- Stratified: fraud rate (3.5%) დაბალძია, stratify=True უზრუნველყოფს თანაბარ განაწილებას
- OOF (Out-of-Fold) predictions: მთლიანი train სეტის AUC-ის შეფასება
- Early stopping: 50 rounds XGBoost-სა და LightGBM-ში overfitting-ის თავიდან ასაცილებლად

### საბოლოო მოდელის შერჩევა

**LightGBM** შეირჩა საუკეთესო individual მოდელად (OOF AUC ~0.928):
1. XGBoost-ზე ოდნავ მაღალი AUC
2. უფრო სწრაფი training large dataset-ზე
3. num_leaves პარამეტრი depth-ზე მეტ კონტროლს იძლევა

**Ensemble** (weighted average by OOF AUC) კიდევ ~0.002-0.003 AUC-ს ამატებს.

---

## MLflow Tracking

### MLflow ექსპერიმენტების ბმული

> DagsHub: `https://dagshub.com/YOUR_DAGSHUB_USERNAME/YOUR_REPO_NAME.mlflow`

### ექსპერიმენტების სტრუქტურა

| ექსპერიმენტი | Run-ები |
|-------------|---------|
| `XGBoost_Training` | XGBoost_Cleaning, XGBoost_Feature_Engineering, XGBoost_Feature_Selection, XGBoost_Underfitted_Baseline, XGBoost_Overfitted, XGBoost_Final_CV, XGBoost_Pipeline_Registry |
| `LightGBM_Training` | LightGBM_Cleaning, LightGBM_Feature_Engineering, LightGBM_Feature_Selection, LightGBM_Underfitted, LightGBM_Overfitted, LightGBM_Final_CV, LightGBM_Pipeline_Registry |
| `CatBoost_Training` | CatBoost_Cleaning, CatBoost_Feature_Engineering, CatBoost_Feature_Selection, CatBoost_Underfitted, CatBoost_Overfitted, CatBoost_Final_CV, CatBoost_Pipeline_Registry |
| `RandomForest_Training` | RF_Cleaning, RF_Feature_Engineering, RF_Feature_Selection, RF_Underfitted, RF_Overfitted, RF_Final_CV, RF_Pipeline_Registry |
| `LogisticRegression_Training` | LR_Cleaning, LR_Feature_Engineering, LR_Feature_Selection, LR_Underfitted, LR_Overfitted, LR_Final_CV, LR_Pipeline_Registry |
| `Inference` | Final_Submission |

### ჩაწერილი მეტრიკები

**Cleaning run-ები**:
- `dropped_high_missing_cols` — ამოღებული სვეტების რაოდენობა
- `train_cols_after_cleaning` — სვეტები გასუფთავების შემდეგ

**Feature Selection run-ები**:
- `features_before` — FS-მდე ფიჩერთა რაოდენობა
- `features_after_importance` — importance filter-ის შემდეგ
- `features_removed_corr` — კორელაციის გამო ამოღებული
- `features_final` — საბოლოო ფიჩერთა რაოდენობა

**Training run-ები**:
- `cv_auc_mean` — Cross-Validation AUC (საშუალო)
- `cv_auc_std` — CV AUC სტანდარტული გადახრა
- `oof_auc` — Out-of-Fold AUC (მთლიანი)
- `fold_1_auc` ... `fold_5_auc` — თითო fold-ის AUC
- `train_auc` — (overfit run-ებში) training AUC
- `overfit_gap` — train_auc - cv_auc (overfitting-ის საზომი)
- `note` — run-ის კომენტარი (e.g. "intentionally_overfitted")

### საუკეთესო მოდელის შედეგები

| მეტრიკა | მნიშვნელობა |
|---------|------------|
| მოდელი | LightGBM (Optuna tuned) |
| OOF AUC | ~0.928 |
| CV AUC Mean | ~0.927 |
| CV AUC Std | ~0.003 |
| Kaggle Public Score | (ატვირთვის შემდეგ) |
| Ensemble AUC | ~0.930 |

---

## შენიშვნები

- ყველა Pipeline შენახულია MLflow Model Registry-ში და პირდაპირ raw test data-ზე მუშაობს
- `model_inference.ipynb` ჩამოტვირთავს საუკეთესო Pipeline-ს Registry-დან OOF AUC-ის მიხედვით
- Kaggle-ზე GPU-ს გამოყენება (`tree_method='gpu_hist'`, `device='gpu'`) training-ს 5-10x აჩქარებს
