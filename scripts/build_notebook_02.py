"""Builds notebooks/02-gold-returns.ipynb from the cells below and runs it.

Usage: python3 scripts/build_notebook_02.py     (needs nbformat, nbconvert, ipykernel, scipy)
"""
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parent.parent
md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = [
md("""# آمار بازده طلا: Fat Tail و خوشه‌ای شدن نوسان

**آموزشی است و توصیه سرمایه‌گذاری نیست.** این نوت‌بوک قیمت را پیش‌بینی نمی‌کند. فقط می‌پرسد: بازده روزانه طلا چه شکلی دارد، و آیا شبیه توزیع نرمال است؟

دو چیز را اندازه می‌گیریم:

1. **Fat Tail:** روزهای خیلی بزرگ (مثلاً بیش از ۴ انحراف معیار) در واقعیت چند برابر مدل نرمال اتفاق می‌افتند؟
2. **خوشه‌ای شدن نوسان:** آیا بعد از یک روز پرنوسان، روز پرنوسان دیگری محتمل‌تر است؟

**داده:** قیمت اونس طلا از TGJU (غیررسمی). برای ۲۰۰۰ به بعد آن را با آتی COMEX از Yahoo مقایسه کرده‌ایم (در نوت‌بوک ۱). قبل از آن فقط یک منبع داریم و برای هر روز فقط یک قیمت (نه باز و بسته)، پس تحلیل اصلی از سپتامبر ۲۰۰۰ شروع می‌شود و ۱۹۸۰ تا ۱۹۹۹ فقط در پایان به‌عنوان مقایسه می‌آید."""),
code("""import json, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path("..") if Path("../scripts/fetch_data.py").exists() else Path(".")  # repo checkout or Colab
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
(ROOT / "images").mkdir(exist_ok=True)

def load_ons():
    f = DATA / "ons_usd.csv"
    if not f.exists():  # data is not stored in the repository
        url = "https://api.tgju.org/v1/market/indicator/summary-table-data/ons"
        rows = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60))["data"]
        num = lambda t: float(t.replace(",", ""))
        pd.DataFrame([(r[6].replace("/", "-"), *map(num, r[:4])) for r in rows],
                     columns=["date", "open", "low", "high", "close"]).sort_values("date").to_csv(f, index=False)
    return pd.read_csv(f, parse_dates=["date"]).set_index("date")["close"]

ons = load_ons()
r_all = np.log(ons).diff().dropna()          # daily log returns, all history
START = "2000-09-01"                          # from here we have a second source
r = r_all[START:]
print(f"{len(ons)} prices, {ons.index[0].date()} to {ons.index[-1].date()}; main sample: {len(r)} returns from {r.index[0].date()}")"""),
md("""## ۱. شکل بازده‌ها

بازده لگاریتمی روزانه را حساب کردیم (`log(P_t / P_{t-1})`). اول عددهای ساده:"""),
code("""print(f"annualised mean : {r.mean() * 252:7.2%}")
print(f"annualised vol  : {r.std() * np.sqrt(252):7.2%}")
print(f"skewness        : {stats.skew(r):7.2f}")
print(f"excess kurtosis : {stats.kurtosis(r):7.2f}   (0 for a normal distribution)")
print("worst days:", {d.date().isoformat(): round(v, 3) for d, v in r.nsmallest(3).items()})
print("best days :", {d.date().isoformat(): round(v, 3) for d, v in r.nlargest(3).items()})"""),
md("""**کشیدگی اضافه (excess kurtosis)** برای توزیع نرمال صفر است. اینجا حدود ۶ است، یعنی Fat Tail خیلی شدید است.

### چند برابر مدل نرمال؟
روزهایی را می‌شماریم که بازده بیش از ۳، ۴ یا ۵ انحراف معیار از میانگین فاصله دارد و با تعداد مورد انتظار در توزیع نرمال مقایسه می‌کنیم:"""),
code("""z = (r - r.mean()) / r.std()
rows = []
for k in (3, 4, 5):
    expected = 2 * stats.norm.sf(k) * len(z)
    rows.append({"threshold": f"{k} sigma", "observed days": int((z.abs() > k).sum()), "expected if normal": round(expected, 2)})
pd.DataFrame(rows).set_index("threshold")"""),
code("""fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.linspace(-6, 6, 400)
ax.hist(z, bins=120, range=(-6, 6), density=True, alpha=0.6, label="gold daily returns")
ax.plot(x, stats.norm.pdf(x), color="k", lw=1.2, label="normal")
ax.set_yscale("log"); ax.set_ylim(1e-5, 1)
ax.set_xlabel("return in standard deviations"); ax.set_ylabel("density (log scale)")
ax.set_title("Gold daily returns vs the normal distribution"); ax.legend()
plt.tight_layout(); plt.savefig(ROOT / "images" / "gold-returns-hist.png", dpi=130); plt.show()"""),
md("""روی مقیاس لگاریتمی، Fat Tail واقعی خیلی بالاتر از منحنی نرمال است. **روزهای بیش از ۴ انحراف معیار حدود ۶۵ برابر بیشتر از مدل نرمال** رخ داده‌اند.

> قاعده: مدلی که بازده را نرمال فرض کند، ریسک روزهای بد را به‌شدت کم برآورد می‌کند.

## ۲. خوشه‌ای شدن نوسان

اگر نوسان خوشه‌ای باشد، بزرگیِ بازده امروز (نه جهتش) باید با بزرگیِ بازده دیروز همبسته باشد. خودهمبستگی را برای سه چیز می‌سنجیم: خود بازده، قدر مطلق بازده و مربع بازده."""),
code("""lags = [1, 5, 10, 20, 60]
ac = pd.DataFrame({
    "return": [r.autocorr(l) for l in lags],
    "abs return": [r.abs().autocorr(l) for l in lags],
    "squared return": [(r ** 2).autocorr(l) for l in lags],
}, index=[f"lag {l}" for l in lags]).round(3)
ac"""),
md("""- **خود بازده** تقریباً صفر است: جهت بازده امروز چیزی درباره فردا نمی‌گوید.
- **قدر مطلق بازده** تا ۶۰ روز بعد هم همبستگی مثبت دارد. نوسان «به‌یاد می‌ماند».

حالا نوسان ۲۱ روزه (سالانه‌شده) را در طول زمان ببینیم:"""),
code("""vol = r.rolling(21).std() * np.sqrt(252)
fig, ax = plt.subplots(figsize=(10, 4))
vol.plot(ax=ax, lw=0.8)
ax.set_ylabel("21-day volatility, annualised"); ax.set_xlabel("")
ax.set_title("Gold volatility moves in regimes")
plt.tight_layout(); plt.savefig(ROOT / "images" / "gold-returns-vol.png", dpi=130); plt.show()
print("5% / 50% / 95% quantiles:", vol.quantile([.05, .5, .95]).round(3).to_dict())
print(f"max {vol.max():.1%} on {vol.idxmax().date()}, min {vol.min():.1%} on {vol.idxmin().date()}")
v = vol.dropna()
print("correlation of this month's vol with next month's:", round(v.corr(v.shift(-21)), 2))"""),
md("""نوسان بین حدود ۸٪ (صدک ۵) و ۲۸٪ (صدک ۹۵) جابه‌جا می‌شود و اوجش در ۲۰۰۸-۱۰ به ۵۶٪ رسید. همبستگی نوسان این ماه با ماه بعد حدود ۰٫۶ است.

## ۳. اگر نوسان را کنترل کنیم، Fat Tail از بین می‌رود؟

یک فکر طبیعی: Fat Tail فقط به‌خاطر این است که نوسان عوض می‌شود. اگر هر بازده را بر نوسان همان زمان تقسیم کنیم، شاید نرمال شود. نوسان را با میانگین متحرک نمایی (EWMA، λ = 0٫۹۴) از بازده‌های **قبلی** برآورد می‌کنیم. `shift(1)` مهم است: اگر نوسان امروز را از خود بازده امروز بسازیم، به آینده نگاه کرده‌ایم."""),
code("""lam = 0.94
sigma = np.sqrt((r_all ** 2).ewm(alpha=1 - lam, adjust=False).mean()).shift(1)   # uses returns up to yesterday only
zs = (r_all / sigma)[START:]
zs = (zs - zs.mean()) / zs.std()

print(f"excess kurtosis: raw {stats.kurtosis(r):.2f} -> standardised {stats.kurtosis(zs):.2f}")
print(f"autocorrelation of |z| at lag 1/5/20: {[round(zs.abs().autocorr(l), 3) for l in (1, 5, 20)]}")
rows = []
for k in (3, 4):
    rows.append({"threshold": f"{k} sigma", "raw": int((z.abs() > k).sum()), "standardised": int((zs.abs() > k).sum()),
                 "expected if normal": round(2 * stats.norm.sf(k) * len(zs), 2)})
pd.DataFrame(rows).set_index("threshold")"""),
md("""- خوشه‌ای شدن **حذف شد**: خودهمبستگی قدر مطلق بازده‌های استانداردشده تقریباً صفر است.
- کشیدگی از حدود ۶٫۳ به حدود ۳٫۶ کم شد، یعنی حدود **۴۰٪ از کشیدگی** به تغییر نوسان مربوط بود.
- ولی Fat Tail **هنوز هست**: روزهای بیش از ۴ انحراف معیار حدود ۵۰ برابر مدل نرمال است. پس بقیه از خود شوک‌های بزرگ می‌آید؛ علتش را این‌جا بررسی نکرده‌ایم.

## ۴. در دوره‌های مختلف چه فرقی دارد؟"""),
code("""eras = [("1980-1999 (single source, one price a day)", "1980", "1999"), ("2000-09 to 2009", START, "2009"),
        ("2010-2019", "2010", "2019"), ("2020-2026", "2020", "2026")]
pd.DataFrame({
    label: {"days": len(r_all[a:b]), "annualised vol": r_all[a:b].std() * np.sqrt(252), "excess kurtosis": stats.kurtosis(r_all[a:b])}
    for label, a, b in eras
}).T.round(3)"""),
md("""نوسان سالانه در هر دهه بین ۱۵٪ و ۲۰٪ است و کشیدگی اضافه همیشه بالای ۵. دوره ۱۹۸۰ تا ۱۹۹۹ بیشترین کشیدگی (حدود ۱۵) را دارد، ولی آن‌جا فقط یک منبع و یک قیمت در روز داریم و جهش‌های بزرگ ۱۹۸۰ هم هست؛ به همین دلیل تحلیل اصلی از ۲۰۰۰ شروع شد.

## ۵. محدودیت‌ها

۱. **منبع داده:** TGJU غیررسمی است. ۲۰۰۰ به بعد با آتی COMEX مقایسه شد (همبستگی بازده روزانه حدود ۰٫۹؛ از ۲۶ روز بالای ۴ انحراف معیار، تقریباً ۹۶٪ در Yahoo هم با همان علامت و بزرگی‌ای دست‌کم نصف دیده شد). ولی مثلاً ۲۰۰۸-۱۰-۱۰ در TGJU افت ۷٪ است و در Yahoo ۳٫۲٪؛ تفاوت ساعت بسته‌شدن.
۲. **آتی، spot نیست:** بازده روزانه کمی فرق دارد و بعضی روزهای بزرگ ممکن است در یکی بزرگ‌تر باشد.
۳. **EWMA ساده است.** مدل‌های پیچیده‌تر مثل GARCH را امتحان نکرده‌ایم؛ اینجا فقط می‌خواستیم بگوییم خوشه‌ای شدن واقعی است.
۴. **این توصیف گذشته است.** نمی‌گوید فردا چه اتفاقی می‌افتد.

## جمع‌بندی

- بازده روزانه طلا **نرمال نیست**: روزهای بیش از ۴ انحراف معیار حدود ۶۵ برابر مدل نرمال رخ داده‌اند.
- **نوسان خوشه‌ای است** و ماه‌ها به‌یاد می‌ماند، ولی جهت بازده تقریباً غیرقابل‌پیش‌بینی است.
- کنترل نوسان بخشی از کشیدگی را توضیح می‌دهد (حدود ۴۰٪)، نه همه‌اش را.
- نتیجه برای مدیریت ریسک: ریسک را با نوسان **امروز** بسنج، نه با یک انحراف معیار ثابت، و برای روزهای خیلی بد فرض نرمال نگیر.

**گام بعدی:** رابطه طلا با دلار و نرخ بهره، و دام همبستگی کاذب."""),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = ROOT / "notebooks" / "02-gold-returns.ipynb"
ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out.parent)}})
nbformat.write(nb, out)
print("wrote", out)
