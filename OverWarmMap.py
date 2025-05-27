import pandas as pd
import numpy as np
import os
import glob

OTHER_TF = r"C:\Users\Accretia_RS\PycharmProjects\pythonProject\otherTF"
WARM_MAPS = r"C:\Users\Accretia_RS\PycharmProjects\pythonProject\WarmMaps"

os.makedirs(WARM_MAPS, exist_ok=True)

h1_files = glob.glob(os.path.join(OTHER_TF, "*_H1.txt"))
d1_files = glob.glob(os.path.join(OTHER_TF, "*_D1.txt"))

tickers = set()
for f in h1_files:
    ticker = os.path.basename(f).replace("_3m_400000_H1.txt", "").replace("_H1.txt", "")
    tickers.add(ticker)
for f in d1_files:
    ticker = os.path.basename(f).replace("_3m_400000_D1.txt", "").replace("_D1.txt", "")
    tickers.add(ticker)

for ticker in sorted(tickers):
    print(f"\n🔵 Обрабатывается тикер: {ticker}")
    # Ищем актуальный файл (поддержка разных вариантов названия)
    h1_file = None
    d1_file = None
    for hf in h1_files:
        if os.path.basename(hf).startswith(ticker):
            h1_file = hf
            break
    for df in d1_files:
        if os.path.basename(df).startswith(ticker):
            d1_file = df
            break
    if not h1_file or not d1_file:
        print(f"  ❌ Пропуск: нет H1 или D1 файла для {ticker}.")
        continue

    # D1 (дневные)
    d1 = pd.read_csv(d1_file, skiprows=1, names=['ticker','per','date','time','open','high','low','close','vol'])
    for col in ['open', 'high', 'low', 'close', 'vol']:
        d1[col] = pd.to_numeric(d1[col], errors='coerce')
    d1 = d1.dropna(subset=['close', 'open', 'high', 'low'])
    d1['datetime'] = pd.to_datetime(
        d1['date'].astype(str).str.zfill(8) + d1['time'].astype(str).str.zfill(6),
        format='%Y%m%d%H%M%S'
    )
    d1 = d1.sort_values('datetime').reset_index(drop=True)
    d1['amplitude'] = (d1['high'] - d1['low'])  # абсолютное движение
    d1['ampl_pct'] = (d1['high'] - d1['low']) / d1['close'] * 100  # в процентах

    # H1 (часовые)
    h1 = pd.read_csv(h1_file, skiprows=1, names=['ticker','per','date','time','open','high','low','close','vol'])
    for col in ['open', 'high', 'low', 'close', 'vol']:
        h1[col] = pd.to_numeric(h1[col], errors='coerce')
    h1 = h1.dropna(subset=['close', 'open', 'high', 'low'])
    h1['datetime'] = pd.to_datetime(
        h1['date'].astype(str).str.zfill(8) + h1['time'].astype(str).str.zfill(6),
        format='%Y%m%d%H%M%S'
    )
    h1 = h1.sort_values('datetime').reset_index(drop=True)
    h1['amplitude'] = (h1['high'] - h1['low'])
    h1['ampl_pct'] = (h1['high'] - h1['low']) / h1['close'] * 100

    # --- Привязка к дню и часу (день начинается в 03:00 МСК)
    h1['hour'] = h1['datetime'].dt.strftime('%H:%M')
    h1['ampl_day'] = (h1['datetime'] - pd.Timedelta(hours=3)).dt.date
    h1['weekday'] = (h1['datetime'] - pd.Timedelta(hours=3)).dt.weekday  # 0=Пн
    h1['weekday_name'] = h1['weekday'].map({0:'Пн',1:'Вт',2:'Ср',3:'Чт',4:'Пт',5:'Сб',6:'Вс'})

    hour_list = [f"{h:02d}:00" for h in list(range(3,24)) + list(range(0,3))]
    pivot = h1.pivot_table(
        index='weekday_name',
        columns='hour',
        values='ampl_pct',  # можно поменять на 'amplitude' для абсолютных значений
        aggfunc='mean'
    ).reindex(index=['Пн','Вт','Ср','Чт','Пт','Сб','Вс'], columns=hour_list)

    # Средняя амплитуда по дням недели (D1)
    d1['weekday'] = (d1['datetime'] - pd.Timedelta(hours=3)).dt.weekday
    d1['weekday_name'] = d1['weekday'].map({0:'Пн',1:'Вт',2:'Ср',3:'Чт',4:'Пт',5:'Сб',6:'Вс'})
    d1_ampl_by_weekday = d1.groupby('weekday_name')['ampl_pct'].mean().reindex(['Пн','Вт','Ср','Чт','Пт','Сб','Вс'])

    # ---------- Сохраняем всё в Excel ----------
    out_excel = os.path.join(WARM_MAPS, f"{ticker}_Amplitude_summary.xlsx")
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        d1[['datetime','open','high','low','close','ampl_pct']].to_excel(writer, sheet_name="D1_Amplitude", index=False)
        d1_ampl_by_weekday.to_frame('mean_ampl_pct').to_excel(writer, sheet_name="D1_by_weekday")
        pivot.to_excel(writer, sheet_name="H1_Amplitude_heatmap")
    print(f"  ✔ Excel-файл с тепловой картой сохранён в {out_excel}")

print("\n🟢🟢🟢 Все тикеры обработаны! 🟢🟢🟢\n")
