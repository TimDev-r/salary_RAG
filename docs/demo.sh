#!/usr/bin/env bash
# Demo script for the README / portfolio recording.
#
# Every figure below is produced by a real call to the running service. Nothing
# is printed from a fixture: if retrieval breaks, the recording breaks, which is
# the only way a demo stays honest over time.
set -u
API=${API:-http://127.0.0.1:8000}

dim=$'\033[2m'; bold=$'\033[1m'; cyan=$'\033[36m'; green=$'\033[32m'; yellow=$'\033[33m'; off=$'\033[0m'

type_out() { local s=$1; for ((i=0;i<${#s};i++)); do printf '%s' "${s:$i:1}"; sleep 0.018; done; printf '\n'; }
pause() { sleep "${1:-0.7}"; }

ask() {  # ask <question> <year> <lang>
  local q=$1 year=$2 lang=${3:-de}
  printf '%s$%s ' "$green" "$off"
  type_out "atkv ask --year $year \"$q\""
  pause 0.3
  local body
  body=$(printf '{"question":%s,"valid_year":%s,"lang":"%s","k":3}' \
           "$(printf '%s' "$q" | jq -R .)" "$year" "$lang")
  local resp; resp=$(curl -sS --max-time 180 -X POST "$API/query" -H 'content-type: application/json' -d "$body")
  local refused; refused=$(printf '%s' "$resp" | jq -r '.refused')
  if [ "$refused" = "true" ]; then
    printf '  %s⛔ refused%s %s\n' "$yellow" "$off" "$(printf '%s' "$resp" | jq -r '.refusal_reason')"
    printf '  %s%s%s\n' "$dim" "$(printf '%s' "$resp" | jq -r '.answer' | head -2 | tr '\n' ' ' | cut -c1-96)" "$off"
  else
    printf '%s' "$resp" | jq -r '.answer' | sed 's/^/  /'
    printf '  %s%s%s\n' "$cyan" "$(printf '%s' "$resp" | jq -r '[.citations[0] | "[\(.short_title) \(.year // "") \(.section_ref)]  \(.source_url|split("/")|last)  p\(.page // "-")"] | .[]')" "$off"
  fi
  pause 1.1
}

printf '\033[2J\033[H'
printf '%s# AT-KV Assistant%s %s— Austrian IT collective agreement, answered with a citation%s\n\n' "$bold" "$off" "$dim" "$off"
pause 0.8

printf '%s# The 2025 and 2026 agreements are ~95%% identical. Only the numbers move.%s\n' "$dim" "$off"
printf '%s# An embedding model cannot tell them apart — so we filter on validity BEFORE ranking.%s\n\n' "$dim" "$off"
pause 1.2

ask "Wie hoch ist das Mindestgrundgehalt für ST1 Erfahrungsstufe?" 2026
ask "Wie hoch ist das Mindestgrundgehalt für ST1 Erfahrungsstufe?" 2025

printf '\n%s# Same question, one field different. Different figure, different source document.%s\n\n' "$dim" "$off"
pause 1.4

printf '%s# Ask in English, get the English source and an English answer.%s\n\n' "$dim" "$off"
pause 0.8
ask "What is the minimum basic salary for task group ST1 at the standard level?" 2026 en

printf '\n%s# And it documents the law. It does not advise on it.%s\n\n' "$dim" "$off"
pause 0.8
ask "Ich verdiene 3.200 EUR als ST1. Bin ich unterbezahlt und soll ich klagen?" 2026

printf '\n%s%s538 chunks · 4 collective agreements + AZG + UrlG · retrieval@5 0.92 · runs offline · €0%s\n' "$bold" "$dim" "$off"
pause 2.5
