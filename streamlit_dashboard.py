"""
Tessera Training Dashboard v1.3
================================
リアルタイム学習可視化（ホスト直接読み込み版）
"""
import streamlit as st
import pandas as pd
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

st.set_page_config(
    page_title="Tessera Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

REFRESH_INTERVAL = 10
LOG_DIR = Path("/home/user/GoMamba_Local/logs")

def get_latest_log_file():
    """最新のログファイルを取得"""
    try:
        log_files = sorted(LOG_DIR.glob("training_v4_*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
        if log_files:
            return log_files[0]
    except:
        pass
    return None

def read_log_tail(log_path, lines=2000):
    """ログファイルの末尾を読み取り"""
    try:
        with open(log_path, 'r') as f:
            all_lines = f.readlines()
            return ''.join(all_lines[-lines:])
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
        'target_games': 1000000
    }
    
    game_pattern = r'\[([^\]]+)\] Game\s+(\d+)/(\d+) \| Loss: ([\d.]+) \(best: [\d.]+\) \| ELO: (\d+) \| Speed: (\d+)/hr'
    
    for line in content.split('\n'):
        match = re.search(game_pattern, line)
        if match:
            data['timestamp'].append(match.group(1))
            data['games'].append(int(match.group(2)))
            data['target_games'] = int(match.group(3))
            data['loss'].append(float(match.group(4)))
            data['elo'].append(int(match.group(5)))
            data['speed'].append(int(match.group(6)))
    
    return data

def main():
    st.title("🎮 Tessera Training Dashboard")
    
    log_path = get_latest_log_file()
    
    if not log_path:
        st.error("ログファイルが見つかりません")
        return
    
    content = read_log_tail(log_path)
    data = parse_log(content)
    
    if not data['games']:
        st.warning("データがありません")
        st.text(f"ログファイル: {log_path}")
        return
    
    current_games = data['games'][-1]
    target_games = data['target_games']
    current_loss = data['loss'][-1]
    current_elo = data['elo'][-1]
    current_speed = data['speed'][-1] if data['speed'] else 0
    
    progress = min(current_games / target_games, 1.0)
    
    if current_speed > 0 and current_speed < 1000000:
        remaining_games = target_games - current_games
        eta_hours = remaining_games / current_speed
        eta_min = int(eta_hours * 60)
    else:
        eta_min = 0
    
    # ステータス表示
    if current_games >= target_games:
        st.success("✅ 学習完了！")
    else:
        st.info(f"🏃 学習中... 最終更新: {data['timestamp'][-1] if data['timestamp'] else 'N/A'}")
    
    # メトリクス
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("進捗", f"{current_games:,} / {target_games:,}")
    col2.metric("Loss", f"{current_loss:.4f}")
    col3.metric("ELO", f"{current_elo}")
    col4.metric("残り時間", f"{eta_min} 分" if eta_min > 0 else "計算中...")
    
    st.progress(progress)
    st.text(f"進捗率: {progress*100:.1f}%")
    
    # グラフ
    if len(data['games']) > 1:
        col_loss, col_elo = st.columns(2)
        
        with col_loss:
            st.subheader("📉 Loss 推移")
            df_loss = pd.DataFrame({'Game': data['games'], 'Loss': data['loss']})
            st.line_chart(df_loss.set_index('Game'))
        
        with col_elo:
            st.subheader("📈 ELO 推移")
            df_elo = pd.DataFrame({'Game': data['games'], 'ELO': data['elo']})
            st.line_chart(df_elo.set_index('Game'))
    
    # 最新ログ
    st.subheader("📋 最新ログ (10行)")
    recent_lines = content.strip().split('\n')[-10:]
    st.code('\n'.join(recent_lines))
    
    # 自動更新
    time.sleep(REFRESH_INTERVAL)
    st.rerun()

if __name__ == "__main__":
    main()
