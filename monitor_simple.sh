#!/bin/bash
WEBHOOK_URL="https://discordapp.com/api/webhooks/1460455078728044719/eltuWcVs1dUYdZOS8XAxs02IalZFvWgVWCBCuQU7elr_PbfxXBE-3geZdqhOLxDzCGdc"

notify() {
    curl -s -H "Content-Type: application/json" -d "{\"content\": \"$1\"}" "$WEBHOOK_URL" > /dev/null
}

cd ~/GoMamba_Local

while true; do
    COUNT=$(docker compose exec -T tessera pgrep -c -f "long_training_v4" 2>/dev/null || echo "0")
    if [ "$COUNT" = "0" ]; then
        notify "🚨 Tessera process died! Check manually."
    fi
    sleep 60
done
