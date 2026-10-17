# gold-research

پژوهش کمی روی طلا، از [Quant Academy](https://quantacademy.ir). **آموزشی است و توصیه سرمایه‌گذاری نیست.**

| نوت‌بوک | موضوع |
|---|---|
| [notebooks/01-coin-bubble.ipynb](notebooks/01-coin-bubble.ipynb) | حباب سکه امامی: ارزش ذاتی، اندازه حباب، و آزمون برگشت |
| [notebooks/02-gold-returns.ipynb](notebooks/02-gold-returns.ipynb) | آمار بازده طلا: Fat Tail و خوشه‌ای شدن نوسان |
| [notebooks/03-gold-dollar-rates.ipynb](notebooks/03-gold-dollar-rates.ipynb) | طلا، دلار و نرخ بهره واقعی: رابطه واقعی یا همبستگی کاذب؟ |
| [notebooks/04-gold-token-xaut.ipynb](notebooks/04-gold-token-xaut.ipynb) | توکن طلا (XAUT) در Bybit: تعقیب قیمت، آخر هفته، پایه و فاندینگ |

## اجرا

```
pip install -r requirements.txt
python3 scripts/fetch_data.py          # داده را از TGJU دوباره می‌گیرد
python3 scripts/verify_data.py         # مقایسه با Yahoo (طلا) و bon-bast (دلار و سکه)
python3 scripts/verify_wayback.py      # دلار و سکه قبل از ۲۰۲۱ با آرشیو اینترنت (حدود ۶ دقیقه)
python3 scripts/build_notebook.py      # نوت‌بوک را می‌سازد و اجرا می‌کند
```

**داده در repo نیست** (شرایط بازنشر منابع نامشخص است). نوت‌بوک خودش آن را از TGJU می‌گیرد؛ `scripts/fetch_data.py` هم همین کار را می‌کند. داده از **TGJU** است (منبع غیررسمی؛ طلا با Yahoo و دلار/سکه از 2013-03 با bon-bast و آرشیو اینترنت روی نمونه تأیید شده؛ آذر 1390 تا اسفند 1391 فقط یک منبع دارد). قیمت اونس به دلار، قیمت دلار و سکه به ریال.
