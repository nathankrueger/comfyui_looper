<#
.SYNOPSIS
    Manage Windows port forwarding rules so WSL2 services are accessible from LAN devices.

.DESCRIPTION
    Creates, removes, or lists netsh portproxy rules and Windows Firewall rules
    to forward traffic from all interfaces to WSL2's internal IP.

.PARAMETER Add
    Add a forwarding rule (requires -Port)

.PARAMETER Remove
    Remove a forwarding rule (requires -Port)

.PARAMETER List
    List all current WSL2 Looper port forwarding rules

.PARAMETER Port
    Port to forward (required for -Add and -Remove)

.EXAMPLE
    .\win_port_forward.ps1 -Add -Port 5000      # Forward port 5000
    .\win_port_forward.ps1 -Add -Port 8080       # Forward port 8080
    .\win_port_forward.ps1 -List                 # List all forwarding rules
    .\win_port_forward.ps1 -Remove -Port 5000    # Remove port 5000 forwarding
#>

param(
    [switch]$Add,
    [switch]$Remove,
    [switch]$List,
    [int]$Port = 0
)

# Validate arguments
$actionCount = ([int]$Add.IsPresent + [int]$Remove.IsPresent + [int]$List.IsPresent)
if ($actionCount -eq 0) {
    Write-Host "Error: specify one of -Add, -Remove, or -List"
    Write-Host "Usage:"
    Write-Host "  .\win_port_forward.ps1 -Add -Port <port>"
    Write-Host "  .\win_port_forward.ps1 -Remove -Port <port>"
    Write-Host "  .\win_port_forward.ps1 -List"
    exit 1
}
if ($actionCount -gt 1) {
    Write-Host "Error: specify only one of -Add, -Remove, or -List"
    exit 1
}
if (($Add -or $Remove) -and $Port -eq 0) {
    Write-Host "Error: -Port is required when using -Add or -Remove"
    exit 1
}

# Ensure running as admin
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Relaunching as Administrator..."
    $args = @("-ExecutionPolicy", "Bypass", "-File", $MyInvocation.MyCommand.Path)
    if ($Add)    { $args += "-Add" }
    if ($Remove) { $args += "-Remove" }
    if ($List)   { $args += "-List" }
    if ($Port -ne 0) { $args += @("-Port", $Port) }
    Start-Process powershell -Verb RunAs -Wait -ArgumentList $args
    exit
}

$ruleName = "WSL2 Port Forward $Port"

if ($List) {
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
} elseif ($Remove) {
    Write-Host "Removing port forward for port $Port..."
    netsh interface portproxy delete v4tov4 listenport=$Port listenaddress=0.0.0.0
    netsh advfirewall firewall delete rule name="$ruleName"
    Write-Host "Done - rules removed for port $Port."
} elseif ($Add) {
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
    Write-Host "Access from your LAN at:"
    Write-Host "  http://<your-windows-ip>:$Port"
    Write-Host ""
    Write-Host "To remove later: .\win_port_forward.ps1 -Remove -Port $Port"
    Write-Host "To list rules:   .\win_port_forward.ps1 -List"
}
