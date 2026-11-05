"""Download ~2 years of daily prices for the watchlist and write data/analysis.json.

Each asset gets trend, momentum, support/resistance and a buy-zone signal with
Thai-language reasons. Runs in GitHub Actions on a schedule.
"""
import json
import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

# (site symbol, Yahoo Finance ticker, currency)
WATCHLIST = [
    ("GOOGL", "GOOGL", "$"), ("INTC", "INTC", "$"), ("DIS", "DIS", "$"), ("META", "META", "$"),
    ("AMD", "AMD", "$"), ("AAPL", "AAPL", "$"), ("MSFT", "MSFT", "$"), ("AMZN", "AMZN", "$"),
    ("NFLX", "NFLX", "$"), ("NVDA", "NVDA", "$"),
    ("KTB", "KTB.BK", "฿"), ("ADVANC", "ADVANC.BK", "฿"), ("KBANK", "KBANK.BK", "฿"), ("PTT", "PTT.BK", "฿"),
    ("PTTGC", "PTTGC.BK", "฿"), ("SCB", "SCB.BK", "฿"), ("BBL", "BBL.BK", "฿"), ("PTTEP", "PTTEP.BK", "฿"),
    ("CPALL", "CPALL.BK", "฿"), ("THAI", "THAI.BK", "฿"), ("STECON", "STECON.BK", "฿"),
    ("XAUUSD", "GC=F", "$"), ("XAGUSD", "SI=F", "$"), ("USDTHB", "THB=X", "฿"),
]

SIGNALS = {
    "accumulate": "โซนสะสม",
    "breakout": "Breakout",
    "rebound": "Oversold เด้งสั้น",
    "overbought": "ร้อนแรง รอย่อ",
    "downtrend": "ขาลง อย่าเพิ่งรับ",
    "neutral": "รอดู",
    "fx": "ค่าเงิน",
}


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, math.nan)
    return 100 - 100 / (1 + rs)


def pct(a: float, b: float) -> float | None:
    if b is None or a is None or b == 0 or math.isnan(a) or math.isnan(b):
        return None
    return round((a / b - 1) * 100, 2)


def num(x, d: int = 2):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), d)


def analyze(df: pd.DataFrame, is_fx: bool = False) -> dict:
    """df has High, Low, Close columns indexed by date, oldest first."""
    df = df.dropna(subset=["Close"])
    c, h, l = df["Close"], df["High"], df["Low"]
    last = float(c.iloc[-1])
    sma20, sma50, sma200 = (c.rolling(n).mean() for n in (20, 50, 200))
    r = rsi(c)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    year = df.iloc[-252:]
    hi52, lo52 = float(year["High"].max()), float(year["Low"].min())
    # Support / resistance from the last 60 sessions, excluding the latest 3 so a fresh move can break them
    win = df.iloc[-63:-3]
    support, resistance = float(win["Low"].min()), float(win["High"].max())

    s50, s200, rsi_now = float(sma50.iloc[-1]), float(sma200.iloc[-1]) if len(c) >= 200 else math.nan, float(r.iloc[-1])
    s200_prev = float(sma200.iloc[-21]) if len(c) >= 221 else math.nan
    up_long = not math.isnan(s200) and last > s200
    s200_rising = not math.isnan(s200_prev) and s200 > s200_prev
    golden = not math.isnan(s200) and s50 > s200
    near_support = (last - support) / support * 100 if support else None
    at_support = near_support is not None and 0 <= near_support < 4
    broke_support = near_support is not None and near_support < 0
    near_s50 = abs(last - s50) / s50 * 100
    dd = (last / hi52 - 1) * 100
    # Long-term uptrend whose price has dipped up to 5% under a still-rising 200-day line
    test_200 = not up_long and s200_rising and last >= s200 * 0.95

    reasons = []
    if not math.isnan(s200):
        slope = "" if math.isnan(s200_prev) else (" ที่ยังชี้ขึ้น" if s200_rising else " ที่กำลังชี้ลง")
        reasons.append(("ยืนเหนือ" if up_long else "อยู่ใต้") + "เส้น 200 วัน" + slope)
    reasons.append(f"RSI {rsi_now:.0f}" + (" Oversold" if rsi_now < 30 else " Overbought" if rsi_now > 70 else ""))
    reasons.append(f"ห่างจุดสูงสุด 52 สัปดาห์ {dd:.1f}%")

    # MACD 12/26/9
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    hist = macd - macd_sig
    above = hist > 0
    crossed = above.iloc[-4:].nunique() > 1  # sign flipped within the last 3 sessions
    if crossed and above.iloc[-1]:
        macd_state = "cross_up"; reasons.append("MACD เพิ่งตัดขึ้นเหนือเส้นสัญญาณ")
    elif crossed:
        macd_state = "cross_down"; reasons.append("MACD เพิ่งตัดลงใต้เส้นสัญญาณ")
    elif above.iloc[-1]:
        macd_state = "bull"
    else:
        macd_state = "bear"
    hist_rising = float(hist.iloc[-1]) > float(hist.iloc[-2])

    # Volume vs 20-day average (skipped for FX, which has none)
    vol_ratio, vol_state = None, None
    if "Volume" in df and df["Volume"].iloc[-25:].sum() > 0:
        v = df["Volume"]
        avg20 = float(v.iloc[-21:-1].mean())
        if avg20 > 0:
            vol_ratio = float(v.iloc[-1]) / avg20
            day_up = float(c.iloc[-1]) >= float(c.iloc[-2])
            if vol_ratio >= 1.5:
                vol_state = "buying" if day_up else "selling"
                reasons.append(f"วอลุ่ม {vol_ratio:.1f} เท่าของค่าเฉลี่ย 20 วัน " + ("แรงซื้อเข้า" if day_up else "แรงขายหนัก"))
            elif vol_ratio < 0.6:
                vol_state = "quiet"

    if broke_support:
        reasons.insert(0, f"หลุดแนวรับ 3 เดือน {support:,.2f} รอยืนกลับก่อน")
    if last > resistance and 50 <= rsi_now <= 75:
        sig = "breakout"; reasons.insert(0, f"ทะลุแนวต้าน 3 เดือน {resistance:,.2f}")
    elif rsi_now > 72:
        sig = "overbought"
    elif test_200 and not broke_support:
        sig = "accumulate"; reasons.insert(0, f"ย่อมาทดสอบเส้น 200 วัน {s200:,.2f} ที่ยังชี้ขึ้น")
    elif up_long and not broke_support and (rsi_now < 45 or near_s50 < 3 or at_support):
        sig = "accumulate"
        if at_support:
            reasons.insert(0, f"ย่อใกล้แนวรับ {support:,.2f} (ห่าง {near_support:.1f}%)")
        elif near_s50 < 3:
            reasons.insert(0, f"ย่อมาใกล้เส้น 50 วัน {s50:,.2f}")
        else:
            reasons.insert(0, "ย่อตัวในแนวโน้มขาขึ้น")
    elif rsi_now < 30:
        sig = "rebound"; reasons.insert(0, "ลงแรงจน Oversold ในแนวโน้มขาขึ้น รอแท่งกลับตัวก่อนเข้า" if up_long else "ลงแรงจน Oversold แต่แนวโน้มยังลง เหมาะแค่เก็งเด้งสั้น")
    elif not math.isnan(s200) and not up_long and not golden:
        sig = "downtrend"
    else:
        sig = "neutral"

    # Score 0-100 for sorting: higher = closer to a sensible entry
    score = 50
    score += 15 if up_long else -15
    score += 10 if s200_rising else -5
    score += 10 if golden else -5
    score += max(-15, min(15, (50 - rsi_now) * 0.6))
    score += 10 if at_support else -8 if broke_support else 0
    score += {"accumulate": 10, "breakout": 8, "rebound": -5, "neutral": 0, "overbought": -10, "downtrend": -15}[sig]
    score += {"cross_up": 6, "bull": 2, "bear": -2, "cross_down": -6}[macd_state] + (2 if hist_rising else -2)
    score += {"buying": 5, "selling": -6}.get(vol_state, 0)
    score = int(max(0, min(100, round(score))))

    if is_fx:
        sig, score = "fx", 0
        reasons = [f"ค่าเงินบาท ใช้ประกอบการซื้อทองและหุ้น US · RSI {rsi_now:.0f}", f"กรอบ 3 เดือน {support:,.2f}–{resistance:,.2f}"]

    closes = c.iloc[-126:]
    return {
        "price": num(last), "date": df.index[-1].strftime("%Y-%m-%d"),
        "chg1d": pct(last, float(c.iloc[-2])), "chg1m": pct(last, float(c.iloc[-22])) if len(c) > 22 else None,
        "chg3m": pct(last, float(c.iloc[-64])) if len(c) > 64 else None, "chg1y": pct(last, float(c.iloc[-253])) if len(c) > 253 else None,
        "rsi": num(rsi_now, 1), "sma20": num(sma20.iloc[-1]), "sma50": num(s50), "sma200": num(s200),
        "hi52": num(hi52), "lo52": num(lo52), "fromHigh": num(dd, 1),
        "support": num(support), "resistance": num(resistance), "atrPct": num(atr.iloc[-1] / last * 100, 2),
        "trend": "up" if up_long and golden else "down" if (not math.isnan(s200) and not up_long and not golden) else "mixed",
        "signal": sig, "signalLabel": SIGNALS[sig], "score": score, "reasons": reasons,
        "spark": [num(x) for x in closes.tolist()],
        "macd": macd_state, "macdHist": num(hist.iloc[-1], 3), "macdRising": hist_rising,
        "volRatio": num(vol_ratio, 2), "volState": vol_state,
    }


STOOQ = {"XAUUSD": "xauusd", "XAGUSD": "xagusd", "USDTHB": "usdthb"}


def from_yahoo(tk: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.Ticker(tk).history(period="2y", interval="1d", auto_adjust=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["High", "Low", "Close", "Volume"]].dropna(subset=["Close"])


def from_stooq(sym: str, tk: str) -> pd.DataFrame:
    code = STOOQ.get(sym) or (tk.lower() + ".us" if "." not in tk and "=" not in tk else None)
    if not code:
        raise ValueError("no stooq symbol")
    df = pd.read_csv(f"https://stooq.com/q/d/l/?s={code}&i=d", parse_dates=["Date"], index_col="Date")
    return df[["High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).iloc[-520:]


def valuation(tk: str) -> dict:
    """Trailing / forward P/E and dividend yield from Yahoo; empty when unavailable."""
    try:
        import yfinance as yf
        info = yf.Ticker(tk).info or {}
    except Exception as e:
        print(f"{tk} valuation failed: {type(e).__name__}")
        return {}
    pe, fpe, dy = info.get("trailingPE"), info.get("forwardPE"), info.get("dividendYield")
    ok = lambda x: isinstance(x, (int, float)) and math.isfinite(x) and 0 < x < 1000
    if ok(dy) and dy < 1:  # some versions return a fraction, others a percent
        dy *= 100
    return {"pe": num(pe, 1) if ok(pe) else None, "fpe": num(fpe, 1) if ok(fpe) else None,
            "divYield": num(dy, 2) if ok(dy) else None}


def headlines(tk: str, n: int = 5) -> list:
    """Latest Yahoo Finance headlines for a ticker (title, url, source, unix time)."""
    import yfinance as yf
    raw = []
    for name, fn in (("ticker", lambda: yf.Ticker(tk).news), ("search", lambda: yf.Search(tk, news_count=n * 2).news)):
        try:
            raw = fn() or []
        except Exception as e:
            print(f"{tk} news via {name} failed: {type(e).__name__}: {e}"[:200])
        if raw:
            break
    if not raw:
        print(f"{tk} news: none")
    elif tk == "AAPL":
        print("news item sample keys:", list(raw[0].keys())[:10], list((raw[0].get("content") or {}).keys())[:12])
    out = []
    for it in raw:
        c = it.get("content") or it
        title = c.get("title")
        url = (c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url") or c.get("link")
        src = (c.get("provider") or {}).get("displayName") or c.get("publisher") or ""
        t = c.get("pubDate") or c.get("providerPublishTime")
        if isinstance(t, str):
            try:
                t = int(datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp())
            except ValueError:
                t = None
        if title and url:
            out.append({"title": title, "url": url, "source": src, "time": t})
        if len(out) >= n:
            break
    return out


def main() -> int:
    import time
    out, failed = [], []
    for sym, tk, cur in WATCHLIST:
        df, errors = None, []
        for name, fn in (("yahoo", lambda: from_yahoo(tk)), ("yahoo-retry", lambda: (time.sleep(3), from_yahoo(tk))[1]),
                         ("stooq", lambda: from_stooq(sym, tk))):
            try:
                df = fn()
                if len(df) >= 60:
                    print(f"{sym}: {len(df)} rows from {name}")
                    break
                errors.append(f"{name}: only {len(df)} rows"); df = None
            except Exception as e:
                errors.append(f"{name}: {type(e).__name__}: {e}"[:200])
        if df is None:
            failed.append(sym); print(f"{sym} FAILED: {errors}"); continue
        try:
            if "=" in tk:  # futures/FX volume jumps at contract rolls, so it is not comparable day to day
                df = df.drop(columns=["Volume"], errors="ignore")
            row = {"sym": sym, "ticker": tk, "cur": cur, **analyze(df, is_fx=sym == "USDTHB")}
            if "=" not in tk:
                row.update(valuation(tk))
            row["news"] = headlines(tk)
            out.append(row)
        except Exception as e:
            failed.append(sym); print(f"{sym} analyze FAILED: {type(e).__name__}: {e}")

    bkk = timezone(timedelta(hours=7))
    payload = {"generated": datetime.now(bkk).strftime("%Y-%m-%d %H:%M"), "tz": "Asia/Bangkok",
               "lookback": "2y", "failed": failed, "items": sorted(out, key=lambda x: (x["signal"] == "fx", -x["score"]))}
    if out:
        path = Path(__file__).resolve().parent.parent / "data" / "analysis.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {len(out)} assets, failed: {failed}")
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
