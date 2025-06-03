import os
import pandas as pd
from datetime import datetime, timedelta

CANDLES = r'C:\Users\777\PycharmProjects\OKX\Candles_400k'
OTHER_TF = r"C:\Users\777\PycharmProjects\pythonProject\otherTF"

os.makedirs(OTHER_TF, exist_ok=True)

def resample_candles(df, tf="1H", day_start_hour=3):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + df['time'].astype(str).str.zfill(6), format='%Y%m%d%H%M%S')
    df = df.sort_values('datetime')

    if tf == "1H":
        df['resample_key'] = df['datetime'].dt.floor('h')
    elif tf == "1D":
        # День начинается с 03:00:00 по Москве (UTC+3)
        df['shifted'] = df['datetime'] - pd.Timedelta(hours=day_start_hour)
        df['resample_key'] = df['shifted'].dt.floor('D') + pd.Timedelta(hours=day_start_hour)
    else:
        raise ValueError("Only 1H or 1D supported")

    grouped = df.groupby('resample_key')
    agg = grouped.agg({
        'ticker': 'first',
        'per': 'first',
        'date': lambda x: x.iloc[0],  # дата первой свечи в группе
        'time': lambda x: x.iloc[0],  # время первой свечи
        'open': lambda x: x.iloc[0],
        'high': 'max',
        'low': 'min',
        'close': lambda x: x.iloc[-1],
        'vol': 'sum',
        'datetime': 'first'
    }).reset_index(drop=True)

    # Дата/время как у первой свечи в группе
    agg['date'] = agg['datetime'].dt.strftime("%Y%m%d")
    agg['time'] = agg['datetime'].dt.strftime("%H%M%S")
    return agg[['ticker','per','date','time','open','high','low','close','vol']]

files = [f for f in os.listdir(CANDLES) if f.endswith('.txt')]
for file in files:
    full_path = os.path.join(CANDLES, file)
    print(f"Обрабатываем файл: {file}")
    # Пропускаем заголовок
    df = pd.read_csv(full_path, skiprows=1, names=['ticker','per','date','time','open','high','low','close','vol'])
    for col in ['open','high','low','close','vol']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # H1
    h1 = resample_candles(df, tf="1H", day_start_hour=3)
    h1_file = os.path.join(OTHER_TF, file.replace(".txt", "_H1.txt"))
    with open(h1_file, "w", encoding="utf-8", newline='') as f:
        f.write("<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>\n")
        h1.to_csv(f, header=False, index=False, lineterminator='\n')
    print(f"  H1 свечи сохранены в {h1_file}")
    # D1 (дневная свеча с началом в 03:00 по Москве)
    d1 = resample_candles(df, tf="1D", day_start_hour=3)
    d1_file = os.path.join(OTHER_TF, file.replace(".txt", "_D1.txt"))
    with open(d1_file, "w", encoding="utf-8", newline='') as f:
        f.write("<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>\n")
        d1.to_csv(f, header=False, index=False, lineterminator='\n')
    print(f"  D1 свечи сохранены в {d1_file}")

print("\nГотово! H1 и D1 файлы для всех тикеров созданы в otherTF.")
