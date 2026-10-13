# Shared viewport parsing. Commas delimit overrides; spaces belong to names.
if [[ -n "${WEBUI_VIEW:-}" ]]; then
  views=("$WEBUI_VIEW")
elif [[ -n "${WEBUI_VIEWS:-}" ]]; then
  IFS=, read -r -a views <<< "$WEBUI_VIEWS"
else
  views=(default_1280-720 default_1366-768 default_1536-864 default_1920-1080 iPad-Pro 'iPhone-13 Pro')
fi
for view in "${views[@]}"; do
  case "$view" in
    default_1280-720|default_1366-768|default_1536-864|default_1920-1080|iPad-Pro|'iPhone-13 Pro') ;;
    *) echo "Invalid WebUI view: '$view'. Separate WEBUI_VIEWS with commas, not spaces." >&2; exit 2 ;;
  esac
done
