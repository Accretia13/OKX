import re
import os
from datetime import datetime
from time import sleep
import pytz
from okx.MarketData import MarketAPI
import httpx

def from_str_ms(ms: str) -> datetime:
    dt_utc = datetime.fromtimestamp(int(ms) / 1000, tz=pytz.UTC)
    dt_msk = dt_utc.astimezone(pytz.timezone("Europe/Moscow"))
    return dt_msk

def get_last_saved_ts(file_name):
    if not os.path.exists(file_name):
        return ""
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in reversed(lines):
        if line and not line.startswith("<"):
            parts = line.strip().split(",")
            if len(parts) >= 4:
                date, time = parts[2], parts[3]
                dt = datetime.strptime(date + time, "%Y%m%d%H%M%S")
                ts = int(dt.replace(tzinfo=pytz.timezone("Europe/Moscow")).timestamp() * 1000)
                return str(ts)
    return ""

def fetch_and_save(inst_id: str, tf="3m", limit=100, total_candles=400000):
    print(f"\n🚀🚀🚀 === STARTING {inst_id} === 🚀🚀🚀\n")
    candles = {}
    client = MarketAPI(flag="0", debug=False)

    ticker = inst_id.replace("-", "")
    per = re.findall(r"\d+", tf)[0]

    # Создаём папку, если её нет
    output_dir = "Candles_400k"
    os.makedirs(output_dir, exist_ok=True)

    # Сохраняем файл в папку
    file_name = os.path.join(output_dir, f"{ticker}_{tf}_{total_candles}.txt")

    after = get_last_saved_ts(file_name)
    loaded_candles = 0
    if after:
        print(f"🔄 [{inst_id}] Resume mode: found file, last after={after}")
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                if line and not line.startswith("<"):
                    parts = line.strip().split(",")
                    if len(parts) >= 9:
                        date, time = parts[2], parts[3]
                        dt = datetime.strptime(date + time, "%Y%m%d%H%M%S")
                        ts = int(dt.replace(tzinfo=pytz.timezone("Europe/Moscow")).timestamp() * 1000)
                        candles[ts] = {
                            "t": dt,
                            "o": float(parts[4]),
                            "h": float(parts[5]),
                            "l": float(parts[6]),
                            "c": float(parts[7]),
                            "v": float(parts[8]),
                        }
        loaded_candles = len(candles)
        print(f"🟢 [{inst_id}] Loaded {loaded_candles} candles from existing file.")

    try:
        page = 0
        while len(candles) < total_candles:
            print(f"🌐 [{inst_id}] Fetching page {page}; after={after or 'none'}; collected={len(candles)} 🔄")

            attempt = 0
            max_attempts = 5
            while attempt < max_attempts:
                try:
                    resp = client.get_history_candlesticks(
                        instId=inst_id,
                        bar=tf,
                        limit=limit,
                        after=after
                    )
                    break
                except (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                    print(f"⚠️ [{inst_id}] Network error: {e}. Sleeping for 10 seconds, attempt {attempt+1}/{max_attempts}...")
                    sleep(10)
                    attempt += 1
            else:
                print(f"❌ [{inst_id}] Failed after {max_attempts} attempts. Saving what is collected so far.")
                break

            data = resp.get("data", [])
            print(f"✅ [{inst_id}] Page {page}: got {len(data)} candles")

            for c in data:
                ts = int(c[0])
                if ts not in candles:
                    candles[ts] = {
                        "t": from_str_ms(c[0]),
                        "o": float(c[1]),
                        "h": float(c[2]),
                        "l": float(c[3]),
                        "c": float(c[4]),
                        "v": float(c[7]),
                    }
            if data:
                after = str(int(data[-1][0]))
            if len(data) < limit:
                print(f"⛔ [{inst_id}] Fewer than limit ({len(data)} < {limit}); no more data.")
                break
            if page and page % 20 == 0:
                print(f"⏸️ [{inst_id}] Safety pause... sleeping for 2 seconds")
                sleep(2)
            page += 1

    except KeyboardInterrupt:
        print(f"⚠️ [{inst_id}] Interrupted by user. Saving collected data and exiting fetch.")

    ts_all = sorted(candles.keys(), reverse=True)
    ts_selected = sorted(ts_all[:total_candles])
    print(f"\n✨ [{inst_id}] Saving {len(ts_selected)} collected candles ✨")

    with open(file_name, "w", encoding="utf-8") as f:
        f.write("<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>\n")
        for ts in ts_selected:
            c = candles[ts]
            date = c["t"].strftime("%Y%m%d")
            time = c["t"].strftime("%H%M%S")
            line = (    
                f"{ticker},{per},{date},{time},"
                f"{c['o']},{c['h']},{c['l']},{c['c']},{c['v']}\n"
            )
            f.write(line)
    print(f"\n🎉 [{inst_id}] Saved {len(ts_selected)} candles to {file_name} 🎉\n")

# Пример вызова
inst_ids = [
        "AEVO-USDT-SWAP",
        "AIDOGE-USDT-SWAP",
        "BAT-USDT-SWAP",
        "BERA-USDT-SWAP",
        "HYPE-USDT-SWAP",
        "WOO-USDT-SWAP",
        "PEOPLE-USDT-SWAP",
        "T-USDT-SWAP",
        "SOON-USDT-SWAP",
        "EIGEN-USDT-SWAP",
        "POL-USDT-SWAP",
        "GMT-USDT-SWAP",
        "ZIL-USDT-SWAP",
        "PARTI-USDT-SWAP",
        "MASK-USDT-SWAP",
        "BCH-USD-SWAP",
        "NEAR-USDT-SWAP",
        "COOKIE-USDT-SWAP",
        "SHELL-USDT-SWAP",
        "ZETA-USDT-SWAP",
        "XRP-USDT-SWAP",
        "HBAR-USDT-SWAP",
        "BNT-USDT-SWAP",
        "ICX-USDT-SWAP",
        "PUFFER-USDT-SWAP",
        "RDNT-USDT-SWAP",
        "AXS-USDT-SWAP",
]

def main():
    for inst_id in inst_ids:
        fetch_and_save(inst_id, tf="3m", limit=100, total_candles=400000)
    print("✅✅✅ All done! ✅✅✅")

if __name__ == "__main__":
    main()

