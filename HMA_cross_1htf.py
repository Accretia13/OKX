import sys
import time
import datetime
import pytz
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

def get_ticker_metrics(ticker):
    """
    Получает метрики для тикера из БД:
      - Изменение цены за 24 часа (рассчитывается как процентное изменение последней свечи
        по сравнению со свечой 96 интервалов назад, т.к. 96 * 15 мин = 24 часа)
      - Изменение цены за 1 час (сравнение с свечой 4 интервала назад)
      - Объём торгов в USDT за 24 часа (сумма volCcyQuote за последние 96 свечей)
      - Объём торгов в USDT за 1 час (сумма volCcyQuote за последние 4 свечи)
    """
    db_path = os.path.join(FOLDER, f"{ticker}.db")
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM candles", conn)
        conn.close()
        if df.empty:
            return None

        # Создаем столбец datetime и сортируем по нему
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
        df = df.sort_values("datetime")
        last_close = df.iloc[-1]["close"]

        # Изменение цены за 24 часа
        if len(df) >= 97:
            close_24h_ago = df.iloc[-97]["close"]
            change_24h = (last_close / close_24h_ago - 1) * 100
        else:
            change_24h = None

        # Изменение цены за 1 час
        if len(df) >= 5:
            close_1h_ago = df.iloc[-5]["close"]
            change_1h = (last_close / close_1h_ago - 1) * 100
        else:
            change_1h = None

        # Объём торгов в USDT за 24 часа (сумма volCcyQuote за последние 96 свечей)
        if len(df) >= 96:
            vol24h = df.iloc[-96:]["volCcyQuote"].sum()
        else:
            vol24h = None

        # Объём торгов в USDT за 1 час (сумма volCcyQuote за последние 4 свечи)
        if len(df) >= 4:
            vol1h = df.iloc[-4:]["volCcyQuote"].sum()
        else:
            vol1h = None

        return {
            "change_24h": change_24h,
            "change_1h": change_1h,
            "vol24h": vol24h,
            "vol1h": vol1h
        }
    except Exception as e:
        print(f"Ошибка при получении метрик для {ticker}: {e}")
        return None

# ------------------------- Настройки и глобальные переменные -------------------------
flag = "0"  # 0 – торговля в продакшене, 1 – демо-режим
marketDataAPI = MarketData.MarketAPI(flag=flag)
moscow_tz = pytz.timezone("Europe/Moscow")
FOLDER = "TICKERS_speaker"
HISTORICAL_LIMIT = 250  # число свечей в базе

# Принимаем параметры для Telegram через аргументы командной строки (значения по умолчанию могут быть заданы)
parser = argparse.ArgumentParser()
parser.add_argument("--telegram_token", default="7886331893:AAHGZK88LdN8kXcosJoPx_L1w9FUNTE5bbI", help="Telegram Bot Token")
# Значение по умолчанию здесь не используется – теперь чат определяется динамически.
parser.add_argument("--telegram_chat_id", default="347332229", help="Telegram Chat ID (будет использоваться только по умолчанию, если бот запущен в режиме одиночного пользователя)")
args = parser.parse_args()
TELEGRAM_TOKEN = args.telegram_token

# Глобальный словарь для отслеживания времени последнего уведомления по тикеру для каждого пользователя
# Ключ: chat_id, значение: словарь {ticker: last_notified_time}
user_last_notified = {}

# Будущий объект цикла событий (будет назначен в main)
event_loop = None

# ------------------------- Функции работы с временем -------------------------
def round_to_nearest_15min(dt):
    """Округляет время до ближайшего 15-минутного интервала"""
    return dt.replace(minute=(dt.minute // 15) * 15, second=0, microsecond=0)

def get_expected_candle_time():
    """Возвращает время последней завершённой 15-минутной свечи (МСК) в формате 'YYYY-MM-DD HH:MM:SS'"""
    now = datetime.datetime.now(pytz.utc).astimezone(moscow_tz)
    last_candle_time = round_to_nearest_15min(now) - datetime.timedelta(minutes=15)
    return last_candle_time.strftime('%Y-%m-%d %H:%M:%S')

def timestamp_to_datetime(timestamp):
    """Переводит метку времени (в мс) в МСК, округлённое до 15 минут"""
    utc_time = datetime.datetime.fromtimestamp(timestamp / 1000, datetime.timezone.utc)
    moscow_time = utc_time.astimezone(moscow_tz)
    rounded_time = round_to_nearest_15min(moscow_time)
    return rounded_time.strftime('%Y-%m-%d %H:%M:%S')

# ------------------------- Функции расчёта индикаторов -------------------------
def calculate_ema(data, period):
    """Вычисляет экспоненциальную скользящую среднюю (EMA)"""
    return data.ewm(span=period, adjust=False).mean()

def wma(series, period):
    """
    Вычисляет взвешенное скользящее среднее (WMA) для pandas.Series,
    где веса линейно возрастают от 1 до period.
    """
    weights = np.arange(1, period + 1)
    return series.rolling(window=period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

def calculate_hma(data, period):
    """
    Вычисляет Hull Moving Average (HMA) для заданного периода с использованием округления:
      - half_period рассчитывается как round(period/2)
      - sqrt_period рассчитывается как round(sqrt(period))
    Формула: HMA = WMA(2*WMA(data, half_period) - WMA(data, period), sqrt_period)
    """
    half_period = int(round(period / 2.0))
    sqrt_period = int(round(np.sqrt(period)))
    wma_half = wma(data, half_period)
    wma_full = wma(data, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    return hma


# ------------------------- Инициализация базы исторических данных -------------------------
async def create_folder(folder_name=FOLDER):
    """Создаёт папку для хранения баз данных или очищает её содержимое."""
    if os.path.exists(folder_name):
        for filename in os.listdir(folder_name):
            file_path = os.path.join(folder_name, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Не удалось удалить {file_path}. Причина: {e}")
    else:
        os.makedirs(folder_name)
    print(f"Папка {folder_name} создана или очищена.")

async def fetch_tickers(threshold=2900000):
    """
    Получает список тикеров по условию:
      - тикер заканчивается на "-USDT-SWAP"
      - произведение vol24h и последней цены больше порога.
    """
    url = "https://www.okx.com/api/v5/market/tickers"
    params = {"instType": "SWAP"}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        async with session.get(url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Ошибка запроса: {response.status}")
            data = await response.json()
    tickers = [ item["instId"] for item in data["data"]
                if item["instId"].endswith("-USDT-SWAP") and
                   float(item.get("vol24h", 0)) * float(item.get("last", 0)) > threshold ]
    print(f"Получено {len(tickers)} тикеров для исторических данных.")
    return tickers

async def fetch_historical_candles(ticker, limit=HISTORICAL_LIMIT):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": ticker, "bar": "1H", "limit": str(limit)}  # Изменено на 1H
    await asyncio.sleep(0.5)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        try:
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    print(f"Ошибка 429 для {ticker}. Жду 10 секунд...")
                    await asyncio.sleep(10)
                    return await fetch_historical_candles(ticker, limit)
                elif response.status != 200:
                    print(f"Ошибка получения свечей для {ticker}: {response.status}")
                    return None
                result = await response.json()
        except aiohttp.ClientError as e:
            print(f"Ошибка запроса для {ticker}: {e}")
            return None
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
            "volCcyQuote": float(candle[7]),
            "confirm": int(candle[8])
        })
    df = pd.DataFrame(rows)
    # Расчет HMA только для часовых данных
    df["hma_9"] = calculate_hma(df["close"], 9)
    df["hma_21"] = calculate_hma(df["close"], 21)
    print(f"Получено {len(df)} свечей для {ticker}")
    return df


async def save_to_sqlite(ticker, df, folder=FOLDER):
    """Сохраняет DataFrame в базу SQLite для тикера."""
    db_path = os.path.join(folder, f"{ticker}.db")
    conn = sqlite3.connect(db_path)
    df.to_sql("candles", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Данные для {ticker} сохранены в {db_path}")

async def initialize_historical_data(tickers):
    """Параллельно загружает исторические данные для всех тикеров и сохраняет их в БД."""
    semaphore = asyncio.Semaphore(5)
    async def limited_fetch(ticker):
        async with semaphore:
            df = await fetch_historical_candles(ticker)
            if df is not None and not df.empty:
                await save_to_sqlite(ticker, df)
    tasks = [limited_fetch(ticker) for ticker in tickers]
    await asyncio.gather(*tasks)

# ------------------------- Проверка сигналов пересечения HMA -------------------------
def check_hma_cross(df):
    """
    Рассчитывает HMA(9) и HMA(21) на часовом таймфрейме (данные уже должны быть часовыми)
    и проверяет последние 4 свечи на наличие пересечения.
    Возвращает 'long' или 'short' при обнаружении пересечения.
    """
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    # Если индикаторы не рассчитаны или нужно пересчитать:
    df["hma_9"] = calculate_hma(df["close"], 9)
    df["hma_21"] = calculate_hma(df["close"], 21)
    
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



async def check_initial_signals(tickers, chat_id):
    """
    Проверяет для каждого тикера, было ли пересечение HMA за последние 2 часа (на 1-часовом таймфрейме).
    Отправляет уведомление (с графиком) пользователю с указанным chat_id.
    """
    # Инициализируем словарь для данного пользователя, если его ещё нет
    if chat_id not in user_last_notified:
        user_last_notified[chat_id] = {}
    for ticker in tickers:
        db_path = os.path.join(FOLDER, f"{ticker}.db")
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql("SELECT * FROM candles", conn)
            conn.close()
        except Exception as e:
            print(f"Ошибка чтения базы для {ticker}: {e}")
            continue
        if df.empty or len(df) < 4:
            continue
        signal = check_hma_cross(df)
        if signal:
            last_candle = df.iloc[-1]
            last_candle_time = f"{last_candle['date']} {last_candle['time']}"
            # Если уведомление ещё не отправлено для последней свечи для данного тикера и пользователя
            if ticker not in user_last_notified[chat_id] or user_last_notified[chat_id][ticker] != last_candle_time:
                user_last_notified[chat_id][ticker] = last_candle_time
                await send_telegram_notification(ticker, signal, chat_id)

# ------------------------- Отправка уведомлений в Telegram -------------------------
def generate_chart(ticker):
    try:
        db_path = os.path.join(FOLDER, f"{ticker}.db")
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM candles", conn)
        conn.close()
        if df.empty:
            return None
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
        df.set_index("datetime", inplace=True)
        # Удаляем строки, где отсутствуют индикаторы
        df = df.dropna(subset=["hma_9", "hma_21"])
        numeric_cols = ["open", "high", "low", "close", "volume", "hma_9", "hma_21"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").ffill().fillna(0)
        add_plots = [
            mpf.make_addplot(df["hma_9"], color="blue", width=1.5),
            mpf.make_addplot(df["hma_21"], color="red", width=1.5)
        ]
        buf = io.BytesIO()
        fig, ax = mpf.plot(
            df,
            type="candle",
            style="charles",
            addplot=add_plots,
            returnfig=True,
            figsize=(12, 8),
            tight_layout=True
        )
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf
    except Exception as e:
        print(f"Ошибка при генерации графика для {ticker}: {e}")
        return None

async def send_telegram_notification(ticker, signal, chat_id):
    image_bytes = generate_chart(ticker)
    if image_bytes is None:
        print(f"{ticker}: не удалось сформировать график")
        return

    # Получаем метрики для тикера
    metrics = get_ticker_metrics(ticker)
    if metrics:
        change_24h = metrics.get("change_24h")
        change_1h = metrics.get("change_1h")
        vol24h = metrics.get("vol24h")
        vol1h = metrics.get("vol1h")
        metrics_text = "\n".join([
            f"Изм.24h:    {change_24h:.2f}%" if change_24h is not None else "Изм. за 24ч: нет данных",
            f"Изм.1h:      {change_1h:.2f}%" if change_1h is not None else "Изм. за 1ч: нет данных",
            f"V__24h:      {vol24h / 1e6:.2f} M USDT" if vol24h is not None else "Объем за 24ч: нет данных",
            f"V___1h:      {vol1h / 1e6:.2f} M USDT" if vol1h is not None else "Объем за 1ч: нет данных"
        ])
    else:
        metrics_text = "Метрики недоступны"

    signal_text = "LONG" if signal == "long" else "SHORT"
    ticker_code = f"<code>{ticker.split('-')[0]}</code>"  # берем только первую часть тикера
    caption = f"Сигнал {signal_text} для        {ticker_code}\n{metrics_text}"
    ticker_url = f"https://www.okx.com/ru/trade-swap/{ticker.lower()}"
    keyboard = {
        "inline_keyboard": [
            [{"text": "GOO!", "url": ticker_url}]
        ]
    }
    telegram_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    data = aiohttp.FormData()
    data.add_field("chat_id", str(chat_id))
    data.add_field("caption", caption)
    data.add_field("parse_mode", "HTML")
    data.add_field("reply_markup", json.dumps(keyboard))
    data.add_field("photo", image_bytes.getvalue(), filename="chart.png", content_type="image/png")
    async with aiohttp.ClientSession() as session:
        async with session.post(telegram_url, data=data) as resp:
            if resp.status != 200:
                print(f"{ticker}: Ошибка отправки уведомления в Telegram для chat_id {chat_id} – {resp.status}")
            else:
                print(f"{ticker}: Уведомление отправлено для chat_id {chat_id}.")

# ------------------------- Функции ожидания команды из Telegram -------------------------
async def send_waiting_message(chat_id):
    """
    Отправляет сообщение с кнопкой "OMG!!!" пользователю с chat_id,
    чтобы тот мог запустить повторный запрос данных и отправку уведомлений.
    """
    keyboard = {
        "inline_keyboard": [
            [{"text": "OMG!!!", "callback_data": "start_work"}]
        ]
    }
    message_text = ("Скрипт находится в режиме ожидания.\n"
                    "Нажмите кнопку ниже, чтобы начать выгрузку данных и отправку уведомлений заново.")
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = aiohttp.FormData()
    data.add_field("chat_id", str(chat_id))
    data.add_field("text", message_text)
    data.add_field("reply_markup", json.dumps(keyboard))
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data) as resp:
            if resp.status != 200:
                print(f"Ошибка отправки сообщения о режиме ожидания для chat_id {chat_id}: {resp.status}")
            else:
                print(f"Сообщение о режиме ожидания отправлено для chat_id {chat_id}.")

async def answer_callback_query(callback_query_id):
    """Отвечает на callback-запрос Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
    payload = {"callback_query_id": callback_query_id, "text": "Запуск работы..."}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                print("Ошибка ответа на callback_query:", resp.status)

async def poll_telegram_updates():
    """
    Опрос обновлений Telegram для обработки команд.
    При получении команды "/start_work", "/omg" или callback "start_work"
    запускается обработка запроса для данного chat_id.
    """
    update_offset = None
    while True:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
        params = {}
        if update_offset:
            params["offset"] = update_offset
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        print("Ошибка опроса Telegram обновлений:", resp.status)
                        await asyncio.sleep(2)
                        continue
                    result = await resp.json()
            except Exception as e:
                print("Исключение при опросе обновлений:", e)
                await asyncio.sleep(2)
                continue
        if "result" in result:
            for update in result["result"]:
                update_offset = update["update_id"] + 1
                if "callback_query" in update:
                    callback_query = update["callback_query"]
                    if callback_query.get("data") == "start_work":
                        await answer_callback_query(callback_query.get("id"))
                        chat_id = callback_query["message"]["chat"]["id"]
                        print(f"Получен callback-запрос на запуск работы от chat_id {chat_id}.")
                        # Запускаем обработку запроса для этого пользователя
                        asyncio.create_task(handle_user_request(chat_id))
                elif "message" in update:
                    message = update["message"]
                    if "text" in message and message["text"].strip().lower() in ["/start_work", "/omg"]:
                        chat_id = message["chat"]["id"]
                        print(f"Получена команда на запуск работы от chat_id {chat_id}.")
                        asyncio.create_task(handle_user_request(chat_id))
        await asyncio.sleep(2)

# ------------------------- Обработка запроса пользователя -------------------------
async def handle_user_request(chat_id):
    """
    Обрабатывает запрос пользователя: очищает базы, загружает данные, проверяет сигналы
    и отправляет уведомления исключительно для данного chat_id.
    """
    # Если необходимо, можно инициализировать отдельное хранилище уведомлений для данного пользователя:
    user_last_notified[chat_id] = {}
    await send_waiting_message(chat_id)
    print(f"Запуск работы для chat_id {chat_id}...")
    await create_folder()
    tickers = await fetch_tickers()
    await initialize_historical_data(tickers)
    await check_initial_signals(tickers, chat_id)
    print(f"Выгрузка данных и уведомления отправлены для chat_id {chat_id}.")
    # После завершения можно отправить сообщение о завершении и ждать нового запроса от этого пользователя
    await send_waiting_message(chat_id)

# ------------------------- Главная функция -------------------------
async def main():
    global event_loop
    event_loop = asyncio.get_running_loop()
    # Запускаем фоновый опрос обновлений Telegram
    asyncio.create_task(poll_telegram_updates())
    print("Бот запущен и ожидает команд от пользователей...")
    # Бесконечный цикл, чтобы программа работала постоянно
    while True:
        await asyncio.sleep(60)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
