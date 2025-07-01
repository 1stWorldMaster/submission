#1. Check if the python3.11 available it not 
#
#   load downlaod folder with deps

#2. check if venv is present or not 
#
# if not load it

#3. check if the model pth is or not if not laod them 
#
#load the model path

#4. run venv and then run the main.py


# set -euo pipefail


# folder_exists() {
#   local folder_path="$1"
#   if [ -d "$folder_path" ]; then
#     echo "Folder exists: $folder_path"
#     return 0
#   else
#     echo "Folder does not exist: $folder_path"
#     return 1
#   fi
# }




# if apt list --installed 2>/dev/null | grep -q "^python3.10-venv/"; then
#     echo "✅ python3.10-venv is installed."
# else
#     echo "❌ python3.10-venv is NOT installed."
# fi


# delete_dir() {
#   local dir_to_check="$1"

#   if [ -d "$dir_to_check" ]; then
#     echo "📂 Directory '$dir_to_check' exists. Deleting..."
#     rm -rf "$dir_to_check"
#     echo "✅ Directory deleted."
#   else
#     echo "❌ Directory '$dir_to_check' does not exist."
#   fi
# }



require_net() {
  ping -c1 -W"${2:-2}" "${1:-8.8.8.8}" >/dev/null 2>&1 && 
    echo "✔ Internet reachable (${1:-8.8.8.8})" || {
    echo "✖ No Internet connection — exiting." >&2
    exit 1
  }
}

check_directory() {
    local dir="$1"
    if [ -d "$dir" ]; then
        echo "✅ Directory '$dir' exists."
        cd python-offline
        sudo apt install ./*.deb
    else
        echo "❌ Directory '$dir' does not exist."
        require_net
        echo "creating file and downloading from net"
        mkdir python-offline
        cd python-offline
        apt-get download python3.11 python3.11-venv
        sudo apt install apt-rdepends
        apt-rdepends python3.11 python3.11-venv python3.11-distutils \
          | grep -v "^ " \
          | xargs apt-get download

        sudo apt install ./*.deb
    fi
}

checkk() {
    local dir="$1"
    if [ -d "$dir" ]; then
        echo "✅ Directory '$dir' exists."
    else
        python3.11 -m venv venv
        source venv/bin/activate
        pip install -r require.txt
    fi
}


isinstalled() {
  if apt list --installed 2>/dev/null | grep -q "^$1/"; then
    echo "✅ $1 is installed."
    checkk
    cd face_recog
    python3.11 main.py
  else
    echo "❌ $1 is NOT installed."
    check_directory python-offline
  fi
}


isinstalled python3.11

