"""
天気取得モジュール - Open-Meteo API（無料・登録不要）
競馬場ごとの天気・馬場状態を自動取得
"""
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional
import json

# 競馬場の緯度経度
RACECOURSE_COORDS = {
    '01': {'name': '札幌', 'lat': 43.04, 'lon': 141.41},
    '02': {'name': '函館', 'lat': 41.82, 'lon': 140.70},
    '03': {'name': '福島', 'lat': 37.74, 'lon': 140.36},
    '04': {'name': '新潟', 'lat': 37.83, 'lon': 139.02},
    '05': {'name': '東京', 'lat': 35.66, 'lon': 139.48},
    '06': {'name': '中山', 'lat': 35.78, 'lon': 140.02},
    '07': {'name': '中京', 'lat': 35.10, 'lon': 137.03},
    '08': {'name': '京都', 'lat': 34.91, 'lon': 135.71},
    '09': {'name': '阪神', 'lat': 34.79, 'lon': 135.36},
    '10': {'name': '小倉', 'lat': 33.87, 'lon': 130.82},
}

# WMOコードから天気への変換
WMO_CODE_TO_WEATHER = {
    0: '晴', 1: '晴', 2: '曇', 3: '曇',
    45: '曇', 48: '曇',
    51: '小雨', 53: '雨', 55: '雨',
    56: '雨', 57: '雨',
    61: '小雨', 63: '雨', 65: '雨',
    66: '雨', 67: '雨',
    71: '小雪', 73: '雪', 75: '雪',
    77: '雪', 80: '雨', 81: '雨', 82: '雨',
    85: '雪', 86: '雪',
    95: '雨', 96: '雨', 99: '雨',
}


def get_weather_from_api(place_code: str, date: str) -> Dict:
    """
    Open-Meteo APIから天気を取得
    
    Args:
        place_code: 場所コード (01-10)
        date: 日付 (YYYY-MM-DD)
    
    Returns:
        {'weather': '晴', 'precipitation': 0.0, 'track_condition': '良'}
    """
    if place_code not in RACECOURSE_COORDS:
        return {'weather': '不明', 'precipitation': 0.0, 'track_condition': '良'}
    
    coord = RACECOURSE_COORDS[place_code]
    
    # 過去データか予報かを判定
    target_date = datetime.strptime(date, '%Y-%m-%d').date()
    today = datetime.now().date()
    
    if target_date <= today:
        # 過去データ
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": coord['lat'],
            "longitude": coord['lon'],
            "start_date": date,
            "end_date": date,
            "daily": "weather_code,precipitation_sum",
            "timezone": "Asia/Tokyo"
        }
    else:
        # 予報データ
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": coord['lat'],
            "longitude": coord['lon'],
            "daily": "weather_code,precipitation_sum",
            "timezone": "Asia/Tokyo"
        }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # 該当日のデータを抽出
        daily = data.get('daily', {})
        dates = daily.get('time', [])
        weather_codes = daily.get('weather_code', [])
        precipitations = daily.get('precipitation_sum', [])
        
        if date in dates:
            idx = dates.index(date)
            weather_code = weather_codes[idx] if idx < len(weather_codes) else 0
            precipitation = precipitations[idx] if idx < len(precipitations) else 0.0
        else:
            weather_code = 0
            precipitation = 0.0
        
        weather = WMO_CODE_TO_WEATHER.get(weather_code, '曇')
        track_condition = estimate_track_condition(precipitation, weather)
        
        return {
            'weather': weather,
            'precipitation': precipitation or 0.0,
            'track_condition': track_condition,
            'place_name': coord['name']
        }
    
    except Exception as e:
        print(f"  ⚠️ 天気取得エラー ({coord['name']}): {e}")
        return {'weather': '不明', 'precipitation': 0.0, 'track_condition': '良'}


def estimate_track_condition(precipitation: float, weather: str) -> str:
    """
    降水量と天気から馬場状態を推定
    
    実際の馬場状態は前日までの雨量も影響するため、
    これは簡易的な推定
    """
    if precipitation is None:
        precipitation = 0.0
    
    if precipitation >= 10.0:
        return '不良'
    elif precipitation >= 5.0:
        return '重'
    elif precipitation >= 2.0:
        return '稍重'
    elif weather in ['雨', '雪']:
        return '稍重'
    else:
        return '良'


def get_weather_for_date(date: str) -> Dict[str, Dict]:
    """
    指定日の全競馬場の天気を取得
    
    Args:
        date: 日付 (YYYY-MM-DD)
    
    Returns:
        {'06': {'weather': '晴', 'track_condition': '良', ...}, ...}
    """
    results = {}
    
    print(f"\n🌤️ 天気情報取得中... ({date})")
    
    for place_code, coord in RACECOURSE_COORDS.items():
        weather_info = get_weather_from_api(place_code, date)
        results[place_code] = weather_info
    
    return results


def display_weather(weather_data: Dict[str, Dict]):
    """天気情報を表示"""
    print("\n" + "="*50)
    print("🌤️ 競馬場別天気・馬場状態")
    print("="*50)
    print(f"{'場所':^8} | {'天気':^6} | {'降水量':^8} | {'馬場':^6}")
    print("-"*50)
    
    for place_code, info in sorted(weather_data.items()):
        name = info.get('place_name', RACECOURSE_COORDS.get(place_code, {}).get('name', '不明'))
        weather = info.get('weather', '不明')
        precip = info.get('precipitation', 0.0)
        track = info.get('track_condition', '良')
        print(f"{name:^8} | {weather:^6} | {precip:>6.1f}mm | {track:^6}")


# テスト実行
if __name__ == "__main__":
    # 今日の天気
    today = datetime.now().strftime('%Y-%m-%d')
    weather_data = get_weather_for_date(today)
    display_weather(weather_data)
    
    # 過去の天気（12月13日）
    print("\n" + "="*50)
    print("📅 12月13日の天気")
    weather_data = get_weather_for_date('2025-12-13')
    display_weather(weather_data)
