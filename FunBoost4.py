import sys
import time
import datetime
import aiohttp
import sqlite3
import concurrent.futures
import pandas as pd
import numpy as np
import okx.MarketData as MarketData
import os
import shutil
import asyncio
import io
import matplotlib.pyplot as plt
import mplfinance as mpf
import json
import argparse
import pytz

# ------------------------- Настройки и глобальные переменные -------------------------
flag = "0"  # 0 – торговля в продакшене, 1 – демо-режим
marketDataAPI = MarketData.MarketAPI(flag=flag)
moscow_tz = pytz.timezone("Europe/Moscow")
FOLDER = "TICKERS_speaker"
HISTORICAL_LIMIT = 100

parser = argparse.ArgumentParser()
parser.add_argument("--telegram_token", default="7465229844:AAF9GuunW1QYiwc72vdXBx2RzZ0JdNldSVs")
parser.add_argument("--telegram_chat_id", default="347332229")
args = parser.parse_args()
TELEGRAM_TOKEN = args.telegram_token

user_last_notified = {}
event_loop = None

# ------------------------- Индикаторы -------------------------
def wma(series, period):
    weights = np.arange(1, period + 1)
    return series.rolling(window=period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

def calculate_hma(data, period):
    half_period = int(round(period / 2.0))
    sqrt_period = int(round(np.sqrt(period)))
    wma_half = wma(data, half_period)
    wma_full = wma(data, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    return hma

# ------------------------- Получение и сохранение исторических данных -------------------------
async def fetch_tickers(threshold=500000):
    top_40 = [
        "BTC", "ETH", "XRP", "BNB", "SOL", "DOGE", "ADA", "TRX",
        "STETH", "WBTC", "SUI", "LINK", "AVAX", "XLM", "SHIB", "HBAR",
        "LEO", "BCH", "TON", "LTC", "DOT", "OKB", "PI", "PEPE",
        "AAVE", "DAI", "UNI", "NEAR", "APT", "JITOSOL", "ONDO", "CRO",
        "ETC", "TRUMP", "ICP", "RENDER", "ATOM", "MATIC", "ARB", "INJ"
    ]
    all_symbols = [f"{symbol}-USDT-SWAP" for symbol in top_40]
    print(f"Используем топ-40 монет: {all_symbols}")
    return all_symbols

async def fetch_historical_candles(ticker, limit=HISTORICAL_LIMIT):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": ticker, "bar": "3m", "limit": str(limit)}
    await asyncio.sleep(0.5)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        async with session.get(url, params=params) as response:
            result = await response.json()
    candles = result.get("data", [])
    if not candles:
        print(f"Нет данных свечей для {ticker}")
        return None
    rows = []
    for candle in reversed(candles):
        ts = int(candle[0])
        dt_obj = datetime.datetime.fromtimestamp(ts / 1000, datetime.timezone.utc).astimezone(moscow_tz)
        rows.append({
            "date": dt_obj.strftime('%Y-%m-%d'),
            "time": dt_obj.strftime('%H:%M:%S'),
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
            "volume": float(candle[5]),
            "volCcy": float(candle[6]),
            "volCcyQuote": float(candle[7])
        })
    df = pd.DataFrame(rows)
    df["hma_9"] = calculate_hma(df["close"], 9)
    df["hma_21"] = calculate_hma(df["close"], 21)
    print(f"{ticker}: загружено {len(df)} свечей")
    return df

async def save_to_sqlite(ticker, df):
    db_path = os.path.join(FOLDER, f"{ticker}.db")
    conn = sqlite3.connect(db_path)
    df.to_sql("candles", conn, if_exists="replace", index=False)
    conn.close()
    print(f"{ticker}: сохранено в базу данных")

async def initialize_historical_data(tickers):
    semaphore = asyncio.Semaphore(5)
    async def limited_fetch(ticker):
        async with semaphore:
            df = await fetch_historical_candles(ticker)
            if df is not None and not df.empty:
                await save_to_sqlite(ticker, df)
    tasks = [limited_fetch(ticker) for ticker in tickers]
    await asyncio.gather(*tasks)

async def update_latest_candles(tickers):
    semaphore = asyncio.Semaphore(10)
    async def limited_update(ticker):
        async with semaphore:
            df = await fetch_historical_candles(ticker)
            if df is not None and not df.empty:
                await save_to_sqlite(ticker, df)
    tasks = [limited_update(ticker) for ticker in tickers]
    await asyncio.gather(*tasks)

# ------------------------- Проверка пересечений HMA -------------------------
def check_hma_cross(df):
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    df = df.dropna(subset=["hma_9", "hma_21"])
    recent = df.tail(4)
    diff = recent["hma_9"] - recent["hma_21"]
    prev = None
    signal = None
    for value in diff:
        if prev is not None:
            if prev < 0 and value > 0:
                signal = "long"
            elif prev > 0 and value < 0:
                signal = "short"
        prev = value
    return signal

# ------------------------- Telegram уведомление -------------------------
def generate_chart(ticker):
    db_path = os.path.join(FOLDER, f"{ticker}.db")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM candles", conn)
    conn.close()
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df.set_index("datetime", inplace=True)
    df = df.dropna(subset=["hma_9", "hma_21"])
    add_plots = [
        mpf.make_addplot(df["hma_9"], color="blue", width=1.5),
        mpf.make_addplot(df["hma_21"], color="red", width=1.5)
    ]
    buf = io.BytesIO()
    fig, ax = mpf.plot(df, type="candle", style="charles", addplot=add_plots, returnfig=True, figsize=(12, 8), tight_layout=True)
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

async def send_telegram_notification(ticker, signal, chat_id):
    print(f"Отправляю сигнал {signal} для {ticker} в Telegram")
    image_bytes = generate_chart(ticker)
    signal_text = "LONG" if signal == "long" else "SHORT"
    caption = f"Сигнал {signal_text} для	 <code>{ticker.split('-')[0]}</code>"
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    keyboard = {"inline_keyboard": [[{"text": "Открыть", "url": f"https://www.okx.com/ru/trade-swap/{ticker.lower()}"}]]}
    data = aiohttp.FormData()
    data.add_field("chat_id", str(chat_id))
    data.add_field("caption", caption)
    data.add_field("parse_mode", "HTML")
    data.add_field("reply_markup", json.dumps(keyboard))
    data.add_field("photo", image_bytes.getvalue(), filename="chart.png", content_type="image/png")
    async with aiohttp.ClientSession() as session:
        await session.post(url, data=data)

# ------------------------- Основная проверка -------------------------
async def check_signals(tickers, chat_id):
    if chat_id not in user_last_notified:
        user_last_notified[chat_id] = {}

    semaphore = asyncio.Semaphore(10)

    async def check_ticker(ticker):
        async with semaphore:
            db_path = os.path.join(FOLDER, f"{ticker}.db")
            if not os.path.exists(db_path):
                print(f"Файл базы данных не найден для {ticker}")
                return
            conn = sqlite3.connect(db_path)
            df = pd.read_sql("SELECT * FROM candles", conn)
            conn.close()
            if df.empty or len(df) < 10:
                print(f"{ticker}: недостаточно данных")
                return
            signal = check_hma_cross(df)
            print(f"{ticker}: сигнал = {signal}")
            if signal:
                last_candle = df.iloc[-1]
                last_time = f"{last_candle['date']} {last_candle['time'] }"
                if ticker not in user_last_notified[chat_id] or user_last_notified[chat_id][ticker] != last_time:
                    user_last_notified[chat_id][ticker] = last_time
                    await send_telegram_notification(ticker, signal, chat_id)

    tasks = [check_ticker(ticker) for ticker in tickers]
    await asyncio.gather(*tasks)

# ------------------------- Запуск -------------------------
async def main():
    print("Скрипт запущен...")
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
    tickers = await fetch_tickers()
    await initialize_historical_data(tickers)
    while True:
        print("Проверяю сигналы...")
        await update_latest_candles(tickers)
        await check_signals(tickers, args.telegram_chat_id)
        await asyncio.sleep(180)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
