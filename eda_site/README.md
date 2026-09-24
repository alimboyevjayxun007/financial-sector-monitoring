# EDA Website (Track A)

Bu papka majburiy EDA veb-saytining manba kodini saqlaydi: statik, offline ishlaydigan
HTML sahifa (`index.html`) + uni generatsiya qilgan Python skripti (`generate_charts.py`)
va unga tegishli grafiklar (`assets/*.png`).

Sayt quyidagi 7 bo'limni qamrab oladi (TASKS.md → Track A):

1. Yondashuvning qisqacha tavsifi
2. Dataset umumiy ko'rinishi va tuzilishi
3. Bir nechta mazmunli EDA vizuallari
4. Tranzaksiya tarixidan asosiy kuzatuvlar
5. Target taqsimoti va xulq-atvor patternlari tahlili
6. EDA'dan kelib chiqqan feature/model g'oyalari
7. Qisqa xulosa

## Fayllar

```
eda_site/
├── index.html            # sayt (to'liq self-contained, tashqi internet/CDN kerak emas)
├── generate_charts.py     # grafiklarni haqiqiy ma'lumotlardan generatsiya qiluvchi skript
├── assets/                # generate_charts.py chiqargan PNG grafiklar (01..08)
└── README.md               # ushbu fayl
```

## Grafiklarni qayta generatsiya qilish

Kerakli kutubxonalar (`pandas`, `numpy`, `pyarrow`, `matplotlib`) loyihaning tizim
python3'ida allaqachon o'rnatilgan — alohida virtualenv yoki `pip install` shart emas.

```bash
cd eda_site
python3 generate_charts.py
```

Skript `fintech_track_data/fintech_data/*.csv` va `*.parquet` fayllarini o'qiydi
(hech narsani o'zgartirmaydi/qayta yozmaydi) va `assets/` papkasiga 8 ta PNG
grafikni qayta yozadi. Terminalda har bir grafik uchun fayl hajmi va bir nechta
tekshirilgan statistikalar (masalan korrelyatsiyalar, target ulushi) chiqadi —
shu orqali natijalarning haqiqiy ma'lumotdan kelib chiqqanini tasdiqlash mumkin.

## Saytni ko'rish

**Lokal:** `eda_site/index.html` faylini brauzerda to'g'ridan-to'g'ri oching
(server yoki build qadam kerak emas — sahifa to'liq self-contained, inline CSS,
tashqi CDN'ga bog'liqlik yo'q).

**GitHub Pages orqali (keyinchalik joylashtirish uchun):**
1. Repozitoriy sozlamalarida **Settings → Pages** bo'limiga o'ting.
2. Source sifatida `main` branch, papka sifatida `/eda_site` (yoki `/` va
   `eda_site/index.html`ni repo root'ga ko'chirish) tanlang.
3. Bir necha daqiqadan so'ng sayt `https://<username>.github.io/<repo>/` (yoki
   `/eda_site/`) manzilida ochiladi.

Muqobil variant: `eda_site/` papkasini Netlify/Vercel'ga drag-and-drop qilish
orqali ham bir zumda public linkka joylashtirish mumkin (build buyrug'i kerak
emas, chunki sahifa allaqachon statik HTML).
