#!/bin/bash
set -e

PORT=0
ACTION=""

usage() {
    cat <<EOF
Usage: $0 -a -p <port>    Add a port forwarding rule
       $0 -r -p <port>    Remove a port forwarding rule
       $0 -l              List all port forwarding rules

Manage Windows port forwarding so WSL2 services are accessible
from other devices on the LAN.

Options:
  -a          Add a forwarding rule (requires -p)
  -r          Remove a forwarding rule (requires -p)
  -l          List all current forwarding rules
  -p <port>   Port to forward (required for -a and -r)
  -h          Show this help

Examples:
  $0 -a -p 5000      # Forward port 5000
  $0 -a -p 8080      # Forward port 8080
  $0 -l              # List all forwarding rules
  $0 -r -p 5000      # Remove port 5000 forwarding
EOF
    exit 1
}

while getopts "arlp:h" opt; do
    case $opt in
        a) ACTION="add" ;;
        r) ACTION="remove" ;;
        l) ACTION="list" ;;
        p) PORT="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [ -z "$ACTION" ]; then
    echo "Error: specify one of -a (add), -r (remove), or -l (list)"
    usage
fi

if [ "$ACTION" != "list" ] && [ "$PORT" -eq 0 ]; then
    echo "Error: -p <port> is required when using -a or -r"
    exit 1
fi

if [ "$ACTION" = "list" ]; then
    # Write a temp PowerShell script
    TEMP_PS1=$(mktemp /tmp/wsl_portfwd_XXXXXX.ps1)
    cat > "$TEMP_PS1" <<'EOPS1'
Write-Host "=== Port Proxy Rules ==="
netsh interface portproxy show v4tov4
Write-Host ""
Write-Host "=== Firewall Rules (WSL2 Looper) ==="
$rules = netsh advfirewall firewall show rule name=all dir=in | Select-String -Pattern "WSL2 Port Forward" -Context 0,4
if ($rules) {
    $rules | ForEach-Object { $_.Line; $_.Context.PostContext }
} else {
    Write-Host "  (none)"
}
Start-Sleep 3
EOPS1

    TEMP_PS1_WIN=$(wslpath -w "$TEMP_PS1")
    powershell.exe -Command "Start-Process powershell -Verb RunAs -Wait -ArgumentList '-ExecutionPolicy Bypass -File \"$TEMP_PS1_WIN\"'"
    rm -f "$TEMP_PS1"

elif [ "$ACTION" = "remove" ]; then
    echo "Removing port forward for port $PORT..."
    TEMP_PS1=$(mktemp /tmp/wsl_portfwd_XXXXXX.ps1)
    cat > "$TEMP_PS1" <<EOPS1
netsh interface portproxy delete v4tov4 listenport=$PORT listenaddress=0.0.0.0
netsh advfirewall firewall delete rule name="WSL2 Port Forward $PORT"
Write-Host "Done - rules removed for port $PORT."
Start-Sleep 3
EOPS1

    TEMP_PS1_WIN=$(wslpath -w "$TEMP_PS1")
    powershell.exe -Command "Start-Process powershell -Verb RunAs -Wait -ArgumentList '-ExecutionPolicy Bypass -File \"$TEMP_PS1_WIN\"'"
    rm -f "$TEMP_PS1"
    echo "Port forward removed for port $PORT."

elif [ "$ACTION" = "add" ]; then
    WSL_IP=$(hostname -I | awk '{print $1}')

    if [ -z "$WSL_IP" ]; then
        echo "Error: Could not detect WSL2 IP address."
        exit 1
    fi

    echo "Setting up port forward:"
    echo "  WSL2 IP:  $WSL_IP"
    echo "  Port:     $PORT"
    echo ""
    echo "A Windows UAC prompt will appear - click Yes to allow."

    TEMP_PS1=$(mktemp /tmp/wsl_portfwd_XXXXXX.ps1)
    cat > "$TEMP_PS1" <<EOPS1
netsh interface portproxy add v4tov4 listenport=$PORT listenaddress=0.0.0.0 connectport=$PORT connectaddress=$WSL_IP
netsh advfirewall firewall add rule name="WSL2 Port Forward $PORT" dir=in action=allow protocol=TCP localport=$PORT
Write-Host "Done - port forward and firewall rule created."
Start-Sleep 3
EOPS1

    TEMP_PS1_WIN=$(wslpath -w "$TEMP_PS1")
    powershell.exe -Command "Start-Process powershell -Verb RunAs -Wait -ArgumentList '-ExecutionPolicy Bypass -File \"$TEMP_PS1_WIN\"'"
    rm -f "$TEMP_PS1"

    echo ""
    echo "Port forward active. Access from your LAN at:"
    echo "  http://<your-windows-ip>:$PORT"
    echo ""
    echo "To remove later: $0 -r -p $PORT"
    echo "To list rules:   $0 -l"
fi
