# WUIT Hackathon — Fintech Track: AML Signal Escalation Scoring

O'zbekiston moliya sektoridagi monitoring bo'limi mijozlarning tranzaksiya tarixidan avtomatik generatsiya qilingan **signal (alert)**larni qabul qiladi. Har bir signalni mutaxassis ko'rib chiqadi va **dismiss (0)** yoki **escalate (1)** qiladi. Loyiha maqsadi — har bir yashirin test signali uchun escalate bo'lish **ehtimolligini (0..1)** bashorat qiluvchi model qurish. Baholash metrikasi: **ROC-AUC**.

To'liq arxitektura va klass diagrammasi uchun: [ARCHITECTURE.md](ARCHITECTURE.md)
Vazifalarning 2 kishiga bo'linishi uchun: [TASKS.md](TASKS.md)

---

## 1. Muammo sababi va yechim (qisqacha)

**Sabab:** signallarning katta qismi yolg'on musbat (train setda 11 595 ta dismiss / 2 405 ta escalate — ya'ni ~17.2% escalate). Xodimlar hammasini birma-bir qo'lda tekshiradi, bu resurs isrofi va real xavfning navbatda qolishiga olib keladi.

**Yechim:** har bir signal uchun uning ortidagi tranzaksiya tarixini (o'rtacha ~500 ta tranzaksiya/signal) sonli xususiyatlarga aylantirib (feature engineering), gradient boosting turidagi ML model bilan escalate ehtimolligini bashorat qilish. EDA shuni ko'rsatdiki, birorta ham yakka xususiyat target bilan kuchli chiziqli bog'liq emas (eng katta |korrelyatsiya| ≈ 0.06) — demak qaror ko'p xususiyatlarning birgalikdagi, nochiziqli ta'siriga bog'liq, shu sababli daraxt-asosli ensemble model (LightGBM / XGBoost / GradientBoosting) tanlandi.

---

## 2. Ma'lumotlar tuzilishi

```
fintech_track_data/fintech_data/
├── train_signals.csv           14 000 qator — signal_id, signal_sanasi, eskalatsiya (target)
├── train_transactions.parquet  6 987 663 qator — shu signallarning tranzaksiya tarixi
├── test_signals.csv            6 000 qator — signal_id, signal_sanasi (target yo'q)
├── test_transactions.parquet   3 027 575 qator — test signallarining tranzaksiya tarixi
└── sample_submission (3).csv   namuna: signal_id, ehtimollik
```

**Ustunlar (o'zbekcha → tushuntirish):**

| Ustun | Tushuntirish |
|---|---|
| `signal_id` | signal (alert) identifikatori |
| `signal_sanasi` | signal yaratilgan sana |
| `eskalatsiya` | target: 1 = escalate, 0 = dismiss |
| `tranzaksiya_vaqti` | tranzaksiya vaqti (timestamp) |
| `kirim_chiqim` | yo'nalish: `kirim` (incoming) / `chiqim` (outgoing) |
| `tranzaksiya_turi` | turi: `karta` (karta), `bank_otkazmasi` (bank o'tkazmasi), `naqd` (naqd pul), `xalqaro` (xalqaro) |
| `miqdor_indeksi` | standartlashtirilgan (z-score turidagi) tranzaksiya summasi indeksi |

**Tekshirilgan sifat xususiyatlari (EDA orqali tasdiqlangan):**
- `signal_id`larda train/test o'rtasida **kesishish yo'q** (0 overlap) — data leakage xavfi yo'q.
- Har bir signalning tranzaksiyasi mavjud (bo'sh tarixli signal yo'q).
- Yo'qolgan (missing) qiymat yo'q.
- Sana oralig'i: signallar 2025-01-01 — 2026-12-31; tranzaksiyalar 2024-07-05 — 2026-12-31 (demak tranzaksiyalar signal sanasidan oldin ham, keyin ham uchraydi — feature yasashda **faqat signal sanasidan oldingi** tranzaksiyalardan foydalanish kerak, aks holda "kelajakni ko'rish" (leakage) xatosi yuzaga keladi).
- Target taqsimoti nomutanosib (imbalanced): ~83% dismiss / ~17% escalate.

---

## 3. Loyiha fayl arxitekturasi

```
WUIT Hackathon/
├── fintech_track_data/fintech_data/   # xom (raw) ma'lumotlar — o'zgartirilmaydi
├── data/processed/                    # generatsiya qilingan feature jadvallari (git'ga qo'shilmaydi)
│   ├── train_features.parquet
│   └── test_features.parquet
├── notebooks/
│   └── submission_pipeline.ipynb      # to'liq, qayta ishlaydigan (reproducible) yakuniy notebook
├── src/
│   ├── config.py                      # yo'llar, sobitlar, feature ro'yxati (umumiy shartnoma)
│   ├── data_loading.py                # DataLoader  (A-track)
│   ├── features.py                    # FeatureBuilder  (A-track)
│   ├── eda.py                         # EDA yordamchi funksiyalar / grafiklar  (A-track)
│   ├── model.py                       # ModelTrainer  (B-track)
│   ├── train.py                       # o'qitish skripti  (B-track)
│   └── predict.py                     # Predictor + SubmissionWriter  (B-track)
├── eda_site/                          # majburiy EDA veb-sayti manba kodi  (A-track)
├── outputs/
│   └── team_<TEAM_ID>.csv             # yakuniy topshiriq fayli
├── tests/                             # yengil sanity testlar
├── ARCHITECTURE.md
├── README.md
├── TASKS.md
└── requirements.txt
```

---

## 4. O'rnatish va ishga tushirish

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Feature jadvalini yasash:

```bash
python3 -m src.features
```

Modelni o'qitish va cross-validation:

```bash
python3 -m src.train
```

Test uchun bashorat va submission faylini yaratish:

```bash
python3 -m src.predict --out "outputs/team_<TEAM_ID>.csv"
```

Yoki butun jarayonni birma-bir ko'rish uchun: `notebooks/submission_pipeline.ipynb` ni oching va tartib bilan ishga tushiring (bu — tekshiruv uchun talab qilinadigan **reproducible notebook**).

Chiqish fayli talablari (majburiy):
- Aynan 2 ustun: `signal_id,ehtimollik`
- Har bir test `signal_id` uchun bitta qator, dublikatsiz, bo'sh qiymatsiz
- `0 <= ehtimollik <= 1`
- Fayl nomi: `team_<TEAM_ID>.csv`

---

## 5. EDA'dan olingan asosiy xulosalar (qisqacha)

- **Nomutanosib target**: ~17.2% escalate → baholashda ROC-AUC (threshold'ga bog'liq bo'lmagan metrika) to'g'ri tanlangan, modelni class-weight/imbalance texnikalari bilan o'qitish kerak.
- **Yakka xususiyat kuchsiz**: eng kuchli korrelyatsiya `amt_max` ≈ -0.06 — signal darajasida oddiy threshold-qoida (masalan "agar summa katta bo'lsa escalate") ishlamaydi.
- **Hajm signal beradi, lekin kuchsiz**: escalate signallarda o'rtacha tranzaksiya soni (523.8) dismiss signallardan (494.0) biroz ko'p, xuddi shunday so'nggi 1/7 kundagi faollik ham biroz yuqoriroq — bu **recency (yaqin vaqtdagi faollik)** xususiyatlarining foydali bo'lishi mumkinligini ko'rsatadi.
- **Yo'nalish va tur aralashmasi deyarli farq qilmaydi**: `frac_kirim`, `frac_karta`, `frac_xalqaro` kabi ulushlar ikkala guruhda deyarli bir xil — bular yakka holda kuchli ajratuvchi emas, lekin boshqa xususiyatlar bilan birga (interaction) foydali bo'lishi mumkin.
- **Xulosa**: signal kuchli, ammo individual ravishda zaif ko'plab xatti-harakat izlaridan yig'iladi → **ko'p xususiyatli agregatsiya + gradient boosting** eng oqilona yondashuv, oddiy qoidalar yoki chiziqli model yetarli emas.

To'liq grafiklar va tahlil `eda_site/` saytida taqdim etiladi.

---

## 6. Jamoaviy ish (GitHub, 2 kishi)

Loyiha ikkita mustaqil trek (A: Data & EDA, B: Modeling & Submission) ga bo'lingan bo'lib, ular faqat `data/processed/*.parquet` fayli "shartnomasi" orqali bog'lanadi — shu sabab ikkala kishi bir-birini kutmasdan parallel ishlashi mumkin. To'liq bo'linish, git-branch strategiyasi va checklist uchun [TASKS.md](TASKS.md) ga qarang.
