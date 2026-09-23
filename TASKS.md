# TASKS — 2 kishiga teng bo'lingan vazifalar

Maqsad: ikki kishi GitHub orqali **bir-birini kutmasdan, parallel** ishlashi. Buning uchun loyiha ikkita mustaqil trekka bo'lingan va ular faqat bitta "shartnoma" fayli (`data/processed/*.parquet`, ustunlari [ARCHITECTURE.md](ARCHITECTURE.md#6-xususiyatlar-shartnomasi-feature-contract) da yozilgan) orqali bog'lanadi.

**Muhim qoida:** ikkalangiz ham ishni boshlashdan oldin bo'lim 6dagi feature-nomlar ro'yxatini o'zgartirmaslikka kelishib oling. Agar yangi feature qo'shmoqchi bo'lsangiz — Slack/chatda bir og'iz aytib qo'yish yetarli, PR review kerak emas (chunki ustunlar additive, eskilarini buzmaydi).

---

## Git strategiyasi

- `main` — faqat ishlaydigan kod
- `track-a/*` — A-shaxsning branchlari (masalan `track-a/feature-engineering`)
- `track-b/*` — B-shaxsning branchlari (masalan `track-b/model-training`)
- Har ikkalasi ham kichik PR'lar orqali `main`ga qo'shiladi, review — bir-biringizga
- B-shaxs A-shaxsning haqiqiy `data/processed/*.parquet` faylini kutmasdan ishlashi uchun, `src/config.py` ichida **mock/dummy feature generator** beriladi (pastda ko'rsatilgan) — shu bilan ishni kuningdan boshlash mumkin, keyin faqat haqiqiy fayl bilan almashtiriladi.

---

## TRACK A — Data, Feature Engineering & EDA (≈50%)

**Egallaydigan fayllar:** `src/data_loading.py`, `src/features.py`, `src/eda.py`, `eda_site/**`, `data/processed/*` (generatsiya qiladi)

1. **Data Loading (`src/data_loading.py`)**
   - `load_signals(path) -> DataFrame`, `load_transactions(path) -> DataFrame`
   - Sana ustunlarini `datetime`ga o'girish, `signal_id` bo'yicha tekshiruv (nunique, overlap yo'qligi)

2. **Feature Engineering (`src/features.py`)**
   - `build(signals_df, transactions_df) -> DataFrame` — ARCHITECTURE.md 6-bo'limdagi barcha ustunlarni hisoblaydi
   - **Leakage'ga qarshi qoida**: faqat `tranzaksiya_vaqti <= signal_sanasi` bo'lgan qatorlardan foydalaning
   - Qo'shimcha g'oyalar (ixtiyoriy, vaqt qolsa): kunlik/haftalik trend, oxirgi N ta tranzaksiyaning statistikasi, tur bo'yicha entropy, kirim/chiqim nisbat trendi
   - `train_features.parquet` va `test_features.parquet`ni `data/processed/`ga yozadi

3. **EDA va sayt (`src/eda.py` + `eda_site/`)**
   - Grafiklar: target taqsimoti, vaqt bo'yicha faollik, kirim/chiqim, tur bo'yicha taqsimot, summalar taqsimoti, escalate vs dismiss farqlari, signal oldidan faollik
   - Sayt: Streamlit / statik HTML / GitHub Pages — README talab qilgan 7 ta bo'limni albatta qamrab olishi kerak (yondashuv tavsifi, dataset umumiy ko'rinishi, EDA vizuallari, kuzatuvlar, target taqsimoti tahlili, EDA'dan kelib chiqqan feature g'oyalari, xulosa)
   - Saytni jamoat uchun ochiq linkka joylashtirish (Streamlit Cloud / GitHub Pages / Vercel / Netlify)

**Tayyor bo'lish mezoni:** `data/processed/train_features.parquet` va `test_features.parquet` mavjud, ustunlar shartnomaga mos, EDA sayti ishlaydigan public URL'ga ega.

---

## TRACK B — Modeling, Evaluation & Submission (≈50%)

**Egallaydigan fayllar:** `src/model.py`, `src/train.py`, `src/predict.py`, `notebooks/submission_pipeline.ipynb`, `outputs/*`

1. **Boshlash uchun mock feature'lar** — `src/config.py`dagi `make_dummy_features(signals_df)` funksiyasi tasodifiy/oddiy agregatsiya bilan shartnomadagi ustunlarni generatsiya qiladi. Shu bilan Track A tugashini kutmasdan model pipeline'ni qurishingiz, test qilishingiz mumkin.

2. **Model (`src/model.py`)**
   - `train(features_df, target) -> Model` — LightGBM/XGBoost/sklearn GradientBoosting
   - `cross_validate(features_df, target) -> float` — Stratified K-Fold (imbalance uchun), ROC-AUC
   - Class imbalance uchun `class_weight` yoki `scale_pos_weight` bilan tajriba
   - Hyperparameter tanlash (oddiy grid/optuna, ixtiyoriy)

3. **Train skripti (`src/train.py`)**
   - `data/processed/train_features.parquet`ni o'qiydi (yoki mock, agar hali tayyor bo'lmasa), modelni o'qitadi, CV natijasini bosib chiqaradi, modelni saqlaydi

4. **Predict + Submission (`src/predict.py`)**
   - `predict(model, test_features_df) -> DataFrame` (`signal_id`, `ehtimollik`)
   - `write(predictions_df, out_path)` — 2 ustun, dublikatsiz, `0<=p<=1`, `team_<TEAM_ID>.csv` formatida
   - Submission faylini `sample_submission (3).csv` bilan schema solishtirib tekshiruvchi kichik validatsiya funksiyasi

5. **Yakuniy notebook (`notebooks/submission_pipeline.ipynb`)**
   - A dan B gacha butun pipeline'ni bitta joyda, boshidan oxirigacha qayta ishga tushiriladigan (reproducible) holda yig'ib chiqadi — bu ham majburiy topshiriq

**Tayyor bo'lish mezoni:** `outputs/team_<TEAM_ID>.csv` mavjud, format talablariga 100% mos, CV ROC-AUC natijasi hujjatlashtirilgan, notebook boshidan oxirigacha xatosiz ishlaydi.

---

## Claude Code bilan ishlash tavsiyasi (ixtiyoriy, tezlashtirish uchun)

Har bir kishi o'z trekida ishlaganda, o'z Claude Code sessiyasini **manager** sifatida yuritishi tavsiya etiladi:

- **Manager** (asosiy sessiya): Opus 5, high effort — trek ichidagi rejalashtirish, fayllar orasidagi bog'liqlikni kuzatish, PR tayyorlash
- **Subagentlar** (parallel bajariladigan mayda vazifalar uchun): Sonnet 5, medium effort — masalan:
  - Track A: bitta subagent grafiklar chizsin, boshqa subagent `eda_site` HTML/Streamlit qismini yozsin — ikkalasi parallel
  - Track B: bitta subagent turli model variantlarini (LightGBM vs XGBoost vs sklearn GBM) sinab ko'rsin, boshqasi `predict.py`/validatsiya qismini yozsin

Bu — ixtiyoriy tezlashtirish usuli; asosiy talab shu ikki trekni GitHub orqali mustaqil olib borishdir.

---

## Umumiy (ikkalasiga ham tegishli, kim tez qo'lga olsa)

- `requirements.txt`ni yangilab borish (yangi kutubxona qo'shilsa)
- `tests/`ga yengil sanity testlar (masalan: feature fayl ustunlari to'g'ri, submission format to'g'ri)
- Yakuniy `README.md`/`ARCHITECTURE.md`ni kerak bo'lsa yangilab turish
