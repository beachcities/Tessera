"""
Tessera Training Dashboard
===========================
リアルタイム学習可視化

機能:
- 進捗バー（残り時間を分で表示）
- Loss 推移グラフ
- ELO 推移グラフ
- 最新ログ表示

起動:
    streamlit run streamlit_dashboard.py

Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
import subprocess

# ページ設定
st.set_page_config(
    page_title="Tessera Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自動更新間隔（秒）
REFRESH_INTERVAL = 10


def get_latest_log_file():
    """最新のログファイルを取得"""
    try:
        result = subprocess.run(
            ["docker", "compose", "exec", "-T", "tessera", 
             "ls", "-t", "/app/logs/"],
            capture_output=True, text=True, cwd="/home/user/GoMamba_Local"
        )
        files = result.stdout.strip().split('\n')
        for f in files:
            if f.startswith('training_v4_') and f.endswith('.log'):
                return f"/app/logs/{f}"
    except:
        pass
    return None


def read_log_content(log_path):
    """ログファイルの内容を読み取り"""
    try:
        result = subprocess.run(
            ["docker", "compose", "exec", "-T", "tessera", 
             "cat", log_path],
            capture_output=True, text=True, cwd="/home/user/GoMamba_Local"
        )
        return result.stdout
    except:
        return ""


def parse_log(content):
    """ログをパースしてデータを抽出"""
    data = {
        'games': [],
        'loss': [],
        'elo': [],
        'speed': [],
        'timestamp': [],
        'target_games': 200000
    }
    
    # ゲームログのパターン
    game_pattern = r'\[([^\]]+)\] Game\s+(\d+)/(\d+) \| Loss: ([\d.]+) \(best: [\d.]+\) \| ELO: (\d+) \| Speed: (\d+)/hr'
    
    for match in re.finditer(game_pattern, content):
        timestamp_str, games, target, loss, elo, speed = match.groups()
        data['games'].append(int(games))
        data['loss'].append(float(loss))
        data['elo'].append(int(elo))
        data['speed'].append(int(speed))
        data['timestamp'].append(timestamp_str)
        data['target_games'] = int(target)
    
    return data


def calculate_eta(current_games, target_games, recent_speeds):
    """直近のSpeedからETAを計算（分単位）"""
    if not recent_speeds or len(recent_speeds) == 0:
        return None, None
    
    # 直近10個のSpeedの平均を使用
    avg_speed = sum(recent_speeds[-10:]) / len(recent_speeds[-10:])
    
    if avg_speed <= 0:
        return None, None
    
    remaining_games = target_games - current_games
    remaining_hours = remaining_games / avg_speed
    remaining_minutes = int(remaining_hours * 60)
    
    finish_time = datetime.now() + timedelta(minutes=remaining_minutes)
    
    return remaining_minutes, finish_time


def main():
    st.title("🎮 Tessera Training Dashboard")
    
    # プレースホルダーを作成
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    metrics_placeholder = st.empty()
    charts_placeholder = st.empty()
    log_placeholder = st.empty()
    
    # 自動更新ループ
    while True:
        log_path = get_latest_log_file()
        
        if not log_path:
            status_placeholder.error("⚠️ ログファイルが見つかりません")
            time.sleep(REFRESH_INTERVAL)
            continue
        
        content = read_log_content(log_path)
        data = parse_log(content)
        
        if not data['games']:
            status_placeholder.warning("⏳ データを読み込み中...")
            time.sleep(REFRESH_INTERVAL)
            continue
        
        # 最新の値
        current_games = data['games'][-1]
        target_games = data['target_games']
        current_loss = data['loss'][-1]
        current_elo = data['elo'][-1]
        current_speed = data['speed'][-1]
        
        # ETA計算
        eta_minutes, finish_time = calculate_eta(
            current_games, target_games, data['speed']
        )
        
        # ステータス表示
        with status_placeholder.container():
            if current_games >= target_games:
                st.success("✅ 学習完了！")
            else:
                st.info(f"🏃 学習中... 最終更新: {datetime.now().strftime('%H:%M:%S')}")
        
        # 進捗バー
        with progress_placeholder.container():
            progress = current_games / target_games
            st.progress(progress)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("進捗", f"{current_games:,} / {target_games:,}")
            with col2:
                if eta_minutes is not None:
                    st.metric("残り時間", f"{eta_minutes} 分")
                else:
                    st.metric("残り時間", "計算中...")
            with col3:
                if finish_time is not None:
                    st.metric("完了予定", finish_time.strftime("%H:%M"))
                else:
                    st.metric("完了予定", "計算中...")
        
        # メトリクス
        with metrics_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Loss (recent)", f"{current_loss:.4f}")
            with col2:
                st.metric("ELO", f"{current_elo}")
            with col3:
                st.metric("Speed", f"{current_speed:,}/hr")
            with col4:
                progress_pct = progress * 100
                st.metric("進捗率", f"{progress_pct:.1f}%")
        
        # グラフ
        with charts_placeholder.container():
            if len(data['games']) > 1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📉 Loss 推移")
                    df_loss = pd.DataFrame({
                        'Game': data['games'],
                        'Loss': data['loss']
                    })
                    st.line_chart(df_loss.set_index('Game'))
                
                with col2:
                    st.subheader("📈 ELO 推移")
                    df_elo = pd.DataFrame({
                        'Game': data['games'],
                        'ELO': data['elo']
                    })
                    st.line_chart(df_elo.set_index('Game'))
        
        # 最新ログ
        with log_placeholder.container():
            st.subheader("📋 最新ログ (10行)")
            lines = content.strip().split('\n')
            recent_lines = lines[-10:] if len(lines) >= 10 else lines
            st.code('\n'.join(recent_lines), language='text')
        
        # 学習完了チェック
        if current_games >= target_games:
            st.balloons()
            break
        
        # 更新間隔
        time.sleep(REFRESH_INTERVAL)


if __name__ == "__main__":
    main()
