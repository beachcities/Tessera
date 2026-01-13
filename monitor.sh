#!/bin/bash
#
# Tessera Monitor v1.0
# ====================
# - プロセス監視（60秒毎）
# - 異常終了時にDiscord通知
# - 最新チェックポイントから自動再起動
#
# Usage:
#   chmod +x monitor.sh
#   ./monitor.sh &
#
# Stop:
#   pkill -f monitor.sh
#

WEBHOOK_URL="https://discordapp.com/api/webhooks/1460455078728044719/eltuWcVs1dUYdZOS8XAxs02IalZFvWgVWCBCuQU7elr_PbfxXBE-3geZdqhOLxDzCGdc"
CHECK_INTERVAL=60
RESTART_DELAY=30

notify_discord() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    curl -s -H "Content-Type: application/json" \
         -d "{\"content\": \"🚨 **Tessera Alert** [$timestamp]\\n$message\"}" \
         "$WEBHOOK_URL" > /dev/null
}

notify_discord_success() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    curl -s -H "Content-Type: application/json" \
         -d "{\"content\": \"✅ **Tessera** [$timestamp]\\n$message\"}" \
         "$WEBHOOK_URL" > /dev/null
}

get_latest_checkpoint() {
    docker compose exec -T tessera ls -t /app/checkpoints/*.pth 2>/dev/null | head -1 | tr -d '\r\n'
}

get_latest_log() {
    docker compose exec -T tessera ls -t /app/logs/training_v4_*.log 2>/dev/null | head -1 | tr -d '\r\n'
}

check_for_errors() {
    local log_file=$(get_latest_log)
    if [ -n "$log_file" ]; then
        docker compose exec -T tessera tail -5 "$log_file" 2>/dev/null | grep -q "❌ Error"
        return $?
    fi
    return 1
}

is_process_running() {
    docker compose exec -T tessera pgrep -f "long_training_v4" > /dev/null 2>&1
    return $?
}

restart_training() {
    local checkpoint="$1"
    echo "[$(date)] Restarting training from: $checkpoint"
    docker compose exec -d tessera bash -c "cd /app && python3.10 src/long_training_v4.py --resume $checkpoint"
}

# メイン監視ループ
echo "=========================================="
echo "🔍 Tessera Monitor Started"
echo "   Check interval: ${CHECK_INTERVAL}s"
echo "   Webhook: configured"
echo "=========================================="

notify_discord_success "Monitor started. Watching for process failures..."

CONSECUTIVE_FAILURES=0
MAX_CONSECUTIVE_FAILURES=5

while true; do
    if ! is_process_running; then
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        
        echo "[$(date)] ⚠️ Process not running (failure #$CONSECUTIVE_FAILURES)"
        
        # 最新チェックポイントを取得
        LATEST_CKPT=$(get_latest_checkpoint)
        
        if [ -z "$LATEST_CKPT" ]; then
            notify_discord "Process died but no checkpoint found! Manual intervention required."
            echo "[$(date)] ❌ No checkpoint found!"
        else
            # エラーログを確認
            LOG_FILE=$(get_latest_log)
            ERROR_MSG=""
            if [ -n "$LOG_FILE" ]; then
                ERROR_MSG=$(docker compose exec -T tessera tail -3 "$LOG_FILE" 2>/dev/null | grep -E "Error|OOM|memory" | head -1 | tr -d '\r\n')
            fi
            
            if [ $CONSECUTIVE_FAILURES -ge $MAX_CONSECUTIVE_FAILURES ]; then
                notify_discord "Process died $CONSECUTIVE_FAILURES times consecutively! Stopping auto-restart.\\nLast error: $ERROR_MSG\\nCheckpoint: $(basename $LATEST_CKPT)"
                echo "[$(date)] ❌ Too many failures, stopping monitor"
                exit 1
            fi
            
            notify_discord "Process died! Restarting...\\nError: $ERROR_MSG\\nCheckpoint: $(basename $LATEST_CKPT)"
            
            # 再起動前に少し待つ
            sleep $RESTART_DELAY
            
            restart_training "$LATEST_CKPT"
            
            # 再起動後、プロセスが立ち上がるのを待つ
            sleep 10
            
            if is_process_running; then
                notify_discord_success "Restart successful! Training resumed from $(basename $LATEST_CKPT)"
                echo "[$(date)] ✅ Restart successful"
            else
                echo "[$(date)] ⚠️ Restart may have failed"
            fi
        fi
    else
        # プロセスが動いている場合、連続失敗カウントをリセット
        if [ $CONSECUTIVE_FAILURES -gt 0 ]; then
            echo "[$(date)] ✅ Process recovered, resetting failure count"
            CONSECUTIVE_FAILURES=0
        fi
    fi
    
    sleep $CHECK_INTERVAL
done
