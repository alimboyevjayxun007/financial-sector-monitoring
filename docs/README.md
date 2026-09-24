# EDA Website (Track A)

Bu papka majburiy EDA veb-saytining manba kodini saqlaydi: statik, offline ishlaydigan
HTML sahifa (`index.html`) + uni generatsiya qilgan Python skripti (`generate_charts.py`)
va unga tegishli grafiklar (`assets/*.png`).

Papka ataylab `docs/` deb nomlangan (`eda_site/` emas) — GitHub Pages faqat `/ (root)`
yoki `/docs` papkasini source sifatida tanlashga ruxsat beradi, shu sabab bu joylashuv
hech qanday qo'shimcha nusxalashsiz to'g'ridan-to'g'ri publish qilinadi.

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
docs/
├── index.html            # sayt (to'liq self-contained, tashqi internet/CDN kerak emas)
├── generate_charts.py    # grafiklarni haqiqiy ma'lumotlardan generatsiya qiluvchi skript
├── assets/                # generate_charts.py chiqargan PNG grafiklar (01..08)
└── README.md              # ushbu fayl
```

## Grafiklarni qayta generatsiya qilish

Kerakli kutubxonalar (`pandas`, `numpy`, `pyarrow`, `matplotlib`) loyihaning tizim
python3'ida allaqachon o'rnatilgan — alohida virtualenv yoki `pip install` shart emas.

```bash
cd docs
python3 generate_charts.py
```

Skript `fintech_track_data/fintech_data/*.csv` va `*.parquet` fayllarini o'qiydi
(hech narsani o'zgartirmaydi/qayta yozmaydi) va `assets/` papkasiga 8 ta PNG
grafikni qayta yozadi. Terminalda har bir grafik uchun fayl hajmi va bir nechta
tekshirilgan statistikalar (masalan korrelyatsiyalar, target ulushi) chiqadi —
shu orqali natijalarning haqiqiy ma'lumotdan kelib chiqqanini tasdiqlash mumkin.

## Saytni ko'rish

**Lokal:** `docs/index.html` faylini brauzerda to'g'ridan-to'g'ri oching
(server yoki build qadam kerak emas — sahifa to'liq self-contained, inline CSS,
tashqi CDN'ga bog'liqlik yo'q).

**GitHub Pages orqali (repo egasi bir marta yoqadi, keyin doim ishlaydi):**
1. Repozitoriy sahifasida **Settings → Pages** bo'limiga o'ting.
2. "Build and deployment" → Source: **Deploy from a branch**.
3. Branch: **main**, papka: **/docs** — Save bosing.
4. Bir necha daqiqadan so'ng sayt `https://<username>.github.io/<repo>/` manzilida
   ochiladi (GitHub bu URL'ni Pages sozlamalari sahifasida ham ko'rsatadi).

Muqobil variant: `docs/` papkasini Netlify/Vercel'ga drag-and-drop qilish orqali ham
bir zumda public linkka joylashtirish mumkin (build buyrug'i kerak emas, chunki sahifa
allaqachon statik HTML).
