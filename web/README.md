# AgroAI Web

AgroAI veb-klienti — Next.js 14 + TypeScript + Tailwind CSS. Bitta umumiy
AgroAI API/Core bilan ishlaydi (Telegram bot ham shu core'dan foydalanadi).

## Ishga tushirish (lokal)

```bash
cd web
npm install
npm run dev        # http://localhost:3000
```

API manzili: `NEXT_PUBLIC_AGROAI_API_URL` (standart: `http://127.0.0.1:8080`).

API'ni alohida terminalda ishga tushiring:

```bash
# loyiha ildizidan
.venv/bin/python -m agroai.api.main     # http://127.0.0.1:8080
```

## Holat

* Suhbatlar (tarix, o'chirish) — amalga oshirilgan (API'da saqlanadi)
* Manba kartochkalari — faqat haqiqiy retrieval manbalari ko'rsatiladi
* Model sozlanmagan bo'lsa — halol holat ko'rsatiladi, soxta javob yo'q
* Vision/video/PDF — hali sozlanmagan (soxta emas)
