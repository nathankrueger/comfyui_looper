<#
.SYNOPSIS
    Set up Windows port forwarding so the looper web UI is accessible from LAN devices.

.DESCRIPTION
    Creates (or removes) a netsh portproxy rule and a Windows Firewall rule
    to forward traffic from all interfaces to WSL2's internal IP.

.PARAMETER Port
    Port to forward (default: 5000)

.PARAMETER Remove
    Remove the forwarding rule instead of adding it

.EXAMPLE
    .\win_port_forward.ps1                # Forward port 5000
    .\win_port_forward.ps1 -Port 8080     # Forward port 8080
    .\win_port_forward.ps1 -Remove        # Remove port 5000 forwarding
    .\win_port_forward.ps1 -Remove -Port 8080
#>

param(
    [int]$Port = 5000,
    [switch]$Remove
)

# Ensure running as admin
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Relaunching as Administrator..."
    $args = @("-ExecutionPolicy", "Bypass", "-File", $MyInvocation.MyCommand.Path, "-Port", $Port)
    if ($Remove) { $args += "-Remove" }
    Start-Process powershell -Verb RunAs -Wait -ArgumentList $args
    exit
}

$ruleName = "WSL2 Looper Port $Port"

if ($Remove) {
    Write-Host "Removing port forward for port $Port..."
    netsh interface portproxy delete v4tov4 listenport=$Port listenaddress=0.0.0.0
    netsh advfirewall firewall delete rule name="$ruleName"
    Write-Host "Done - rules removed."
} else {
    # Detect WSL2 IP
    $wslIp = (wsl hostname -I).Trim().Split()[0]
    if (-not $wslIp) {
        Write-Error "Could not detect WSL2 IP address. Is WSL running?"
        exit 1
    }

    Write-Host "Setting up port forward:"
    Write-Host "  WSL2 IP:  $wslIp"
    Write-Host "  Port:     $Port"
    Write-Host ""

    netsh interface portproxy add v4tov4 listenport=$Port listenaddress=0.0.0.0 connectport=$Port connectaddress=$wslIp
    netsh advfirewall firewall add rule name="$ruleName" dir=in action=allow protocol=TCP localport=$Port

    Write-Host ""
    Write-Host "Done - port forward and firewall rule created."
    Write-Host "Access the looper from your LAN at:"
    Write-Host "  http://<your-windows-ip>:$Port"
    Write-Host ""
    Write-Host "To remove later: .\win_port_forward.ps1 -Remove -Port $Port"
}
