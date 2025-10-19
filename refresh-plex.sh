#!/usr/bin/env bash
set -euo pipefail

TOKEN="PLEX_TOKEN_HERE"
BASE="http://127.0.0.1:32400"
SECTIONS=(1 2)

for id in "${SECTIONS[@]}"; do
  printf "▶️  Refrescando sección %s... " "$id"
  code=$(curl -s -o /dev/null -w "%{http_code}" \
    "$BASE/library/sections/$id/refresh?X-Plex-Token=$TOKEN")
  echo "HTTP $code"
done

echo "🔎 Estado (refreshing/scannedAt):"
curl -s "$BASE/library/sections?X-Plex-Token=$TOKEN" | \
  awk -F'"' '/<Directory /{
    for(i=1;i<=NF;i++){
      if($i=="key") k=$(i+2);
      if($i=="title") t=$(i+2);
      if($i=="refreshing") r=$(i+2);
      if($i=="scannedAt") s=$(i+2);
    }
    if(k!=""){printf " - ID %s (%s): refreshing=%s, scannedAt=%s\n", k, t, r, s; k=t=r=s=""}
  }'
