folder_path="python_offline"

if [ -d "$folder_path" ]; then
  echo "Folder exists."
else
  echo "Folder does not exist."
fi
check_net() {
  if ping -q -c 1 -W 2 8.8.8.8 >/dev/null; then
    echo "✅ Internet is available."
  else
    echo "❌ No internet connection."
  fi
}
