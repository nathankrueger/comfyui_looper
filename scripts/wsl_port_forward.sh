#!/bin/bash
set -e

PORT=5000
REMOVE=false

usage() {
    cat <<EOF
Usage: $0 [-p port] [-r]

Set up Windows port forwarding so the looper web UI (running in WSL2)
is accessible from other devices on the LAN.

Options:
  -p <port>   Port to forward (default: $PORT)
  -r          Remove the forwarding rule instead of adding it
  -h          Show this help

Examples:
  $0              # Forward port 5000
  $0 -p 8080      # Forward port 8080
  $0 -r           # Remove port 5000 forwarding
  $0 -r -p 8080   # Remove port 8080 forwarding
EOF
    exit 1
}

while getopts "p:rh" opt; do
    case $opt in
        p) PORT="$OPTARG" ;;
        r) REMOVE=true ;;
        h) usage ;;
        *) usage ;;
    esac
done

WSL_IP=$(hostname -I | awk '{print $1}')

if [ -z "$WSL_IP" ]; then
    echo "Error: Could not detect WSL2 IP address."
    exit 1
fi

# Write a temp PowerShell script to avoid quoting hell between bash and powershell
TEMP_PS1=$(mktemp /tmp/wsl_portfwd_XXXXXX.ps1)

if [ "$REMOVE" = true ]; then
    echo "Removing port forward for port $PORT..."
    cat > "$TEMP_PS1" <<EOPS1
netsh interface portproxy delete v4tov4 listenport=$PORT listenaddress=0.0.0.0
netsh advfirewall firewall delete rule name="WSL2 Looper Port $PORT"
Write-Host "Done - rules removed."
Start-Sleep 3
EOPS1
else
    echo "Setting up port forward:"
    echo "  WSL2 IP:  $WSL_IP"
    echo "  Port:     $PORT"
    echo ""
    echo "A Windows UAC prompt will appear - click Yes to allow."
    cat > "$TEMP_PS1" <<EOPS1
netsh interface portproxy add v4tov4 listenport=$PORT listenaddress=0.0.0.0 connectport=$PORT connectaddress=$WSL_IP
netsh advfirewall firewall add rule name="WSL2 Looper Port $PORT" dir=in action=allow protocol=TCP localport=$PORT
Write-Host "Done - port forward and firewall rule created."
Start-Sleep 3
EOPS1
fi

# Convert to Windows path and run elevated
TEMP_PS1_WIN=$(wslpath -w "$TEMP_PS1")
powershell.exe -Command "Start-Process powershell -Verb RunAs -Wait -ArgumentList '-ExecutionPolicy Bypass -File \"$TEMP_PS1_WIN\"'"
rm -f "$TEMP_PS1"

if [ "$REMOVE" = true ]; then
    echo "Port forward removed."
else
    echo ""
    echo "Port forward active. Access the looper from your LAN at:"
    echo "  http://<your-windows-ip>:$PORT"
    echo ""
    echo "To remove later: $0 -r -p $PORT"
fi
