<#
.SYNOPSIS
Open current MVP evidence and run the maintained development checks.
.DESCRIPTION
This launcher views existing outputs. It never selects a new dataset, records an
annotation approval, starts customer training, or changes experiment protocols.
Report servers bind only to loopback. Stop acts only on the recorded process
started by this launcher, after verifying its start time and executable path.
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('menu', 'status', 'reports', 'scenes', 'predictions', 'annotations', 'check', 'test', 'stop', 'help')]
    [string]$Action = 'menu',
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$localConfigurationPath = Join-Path $projectRoot 'configs\local.json'
if (-not (Test-Path -LiteralPath $localConfigurationPath)) {
    throw 'Create configs/local.json from configs/local.example.json before running the launcher.'
}
$localConfiguration = Get-Content -LiteralPath $localConfigurationPath -Raw | ConvertFrom-Json
$reportPort = 8810
if ($null -ne $localConfiguration.report_workspace.port) {
    $reportPort = $localConfiguration.report_workspace.port
}
if ($reportPort -isnot [int] -or $reportPort -lt 1 -or $reportPort -gt 65535) {
    throw 'report_workspace.port must be an integer from 1 to 65535.'
}
$reportUrl = "http://127.0.0.1:$reportPort"
if (-not [System.IO.Path]::IsPathRooted($localConfiguration.workspace_root)) {
    throw 'workspace_root must be an absolute path.'
}
$workspaceRoot = [System.IO.Path]::GetFullPath($localConfiguration.workspace_root)
$serviceRoot = Join-Path $workspaceRoot 'services'
$sessionPath = Join-Path $serviceRoot 'reports.json'

function Get-ProjectPython {
    $runtimePath = $localConfigurationPath
    if (-not (Test-Path -LiteralPath $runtimePath)) {
        throw 'Missing configs/local.json. Select the project runtime first.'
    }
    $runtime = Get-Content -LiteralPath $runtimePath -Raw | ConvertFrom-Json
    if (-not (Test-Path -LiteralPath $runtime.python_executable -PathType Leaf)) {
        throw 'The configured Python executable is unavailable.'
    }
    return $runtime.python_executable
}

function Open-ProjectUrl([string]$Url) {
    Write-Host "Open: $Url"
    if (-not $NoBrowser) { Start-Process -FilePath $Url }
}

function Get-ReportIdentity {
    try {
        $identity = Invoke-RestMethod -Uri "$reportUrl/mvp/project.json" -TimeoutSec 2
        # Windows PowerShell may return BOM-prefixed JSON as a plain string when
        # a static server omits charset. Accept both normal and legacy decoding.
        if ($identity -is [string]) {
            $prefixCharacters = [char[]]@([char]0xFEFF, [char]0xEF, [char]0xBB, [char]0xBF)
            $identity = $identity.TrimStart($prefixCharacters) | ConvertFrom-Json
        }
        return $identity
    } catch {
        return $null
    }
}

function Test-ReportPort {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $pending = $client.ConnectAsync('127.0.0.1', $reportPort)
        return ($pending.Wait(250) -and $client.Connected)
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Confirm-ReportIdentity($Identity) {
    if ($null -eq $Identity -or $Identity.kind -ne 'dubaidata-mvp-workspace') {
        throw "Port $reportPort belongs to another service. It has been left running."
    }
    if ([System.IO.Path]::GetFullPath($Identity.project_root) -ne $projectRoot) {
        throw 'The report service belongs to another checkout. It has been left running.'
    }
}

function Initialize-ReportWorkspace {
    $hubRoot = Join-Path $workspaceRoot 'reports\mvp'
    $template = Join-Path $PSScriptRoot 'templates\mvp_index.html'
    if (-not (Test-Path -LiteralPath $template -PathType Leaf)) {
        throw 'Missing scripts/templates/mvp_index.html.'
    }
    # Only this generated navigation directory is written. Existing datasets and
    # frozen report payloads remain at their recorded paths.
    New-Item -ItemType Directory -Path $hubRoot -Force | Out-Null
    $pythonExe = Get-ProjectPython
    & $pythonExe -B -m src.validation.report_workspace --configuration $localConfigurationPath --template $template
    if ($LASTEXITCODE -ne 0) { throw 'Report navigation validation failed.' }
    $identity = [pscustomobject]@{
        kind = 'dubaidata-mvp-workspace'
        project_root = $projectRoot
    }
    $utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText((Join-Path $hubRoot 'project.json'), ($identity | ConvertTo-Json), $utf8WithoutBom)
}

function Start-ReportServer {
    if (Test-ReportPort) {
        Confirm-ReportIdentity (Get-ReportIdentity)
        Initialize-ReportWorkspace
        return
    }
    $pythonExe = Get-ProjectPython
    $reports = Join-Path $workspaceRoot 'reports'
    Initialize-ReportWorkspace
    New-Item -ItemType Directory -Path $serviceRoot -Force | Out-Null
    $serverArgs = @('-B', '-m', 'http.server', "$reportPort", '--bind', '127.0.0.1', '--directory', ('"' + $reports + '"'))
    $server = Start-Process -FilePath $pythonExe -ArgumentList $serverArgs -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $serviceRoot 'reports.stdout.log') -RedirectStandardError (Join-Path $serviceRoot 'reports.stderr.log') -PassThru
    $server.Refresh()
    [pscustomobject]@{
        pid = $server.Id
        started_utc = $server.StartTime.ToUniversalTime().ToString('o')
        executable = $server.Path
        project_root = $projectRoot
        port = $reportPort
    } | ConvertTo-Json | Set-Content -LiteralPath $sessionPath -Encoding UTF8
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Milliseconds 200
        $identity = Get-ReportIdentity
        if ($null -ne $identity) {
            Confirm-ReportIdentity $identity
            return
        }
        $server.Refresh()
        if ($server.HasExited) { throw 'Report server stopped. Inspect the workspace services/reports.stderr.log.' }
    }
    throw 'Report server did not become ready. Inspect the workspace services/reports.stderr.log.'
}

function Stop-ReportServer {
    if (-not (Test-Path -LiteralPath $sessionPath)) {
        Write-Host 'No report server was started by this launcher.'
        return
    }
    $session = Get-Content -LiteralPath $sessionPath -Raw | ConvertFrom-Json
    if ($session.project_root -ne $projectRoot -or $session.port -ne $reportPort) {
        throw 'Recorded server does not belong to this workspace.'
    }
    $server = Get-Process -Id $session.pid -ErrorAction SilentlyContinue
    if ($null -eq $server) {
        Write-Host 'The recorded report process has already stopped.'
        return
    }
    if ($server.StartTime.ToUniversalTime().ToString('o') -ne $session.started_utc -or $server.Path -ne $session.executable) {
        throw 'The process ID has been reused. No process was stopped.'
    }
    # The Windows virtualenv launcher can own a child CPython process. Stop this
    # verified process tree rather than leaving an orphaned listening server.
    & taskkill.exe /PID $session.pid /T /F | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'The report process could not be stopped.' }
    Write-Host 'Report server stopped. Saved files are unchanged.'
}

function Get-SelectedReportUrl([string]$Name) {
    $relative = $localConfiguration.report_workspace.$Name
    if ([string]::IsNullOrWhiteSpace($relative)) {
        throw "Select report_workspace.$Name in configs/local.json first."
    }
    # Navigation validation has already checked containment and file existence.
    $encodedSegments = $relative.Split('/') | ForEach-Object { [Uri]::EscapeDataString($_) }
    return "$reportUrl/$($encodedSegments -join '/')"
}

function Show-Menu {
    Write-Host ''
    Write-Host 'DubaiDATA MVP'
    Write-Host '1  Current status'
    Write-Host '2  Results workspace'
    Write-Host '3  Synthetic scenes'
    Write-Host '4  Prediction comparison'
    Write-Host '5  Saved annotations'
    Write-Host '6  Environment check'
    Write-Host '7  Software tests'
    Write-Host '8  Stop the report server'
    Write-Host '0  Exit'
    $choice = Read-Host 'Choose a number'
    $choices = @{'1'='status'; '2'='reports'; '3'='scenes'; '4'='predictions'; '5'='annotations'; '6'='check'; '7'='test'; '8'='stop'; '0'='exit'}
    if (-not $choices.ContainsKey($choice)) { throw 'Choose a number from 0 to 8.' }
    return $choices[$choice]
}

Push-Location $projectRoot
try {
    if ($Action -eq 'menu') {
        $selectedAction = Show-Menu
        if ($selectedAction -eq 'exit') { return }
        $Action = $selectedAction
    }
    switch ($Action) {
        'exit' { return }
        'status' { Get-Content -LiteralPath 'docs\status.md' }
        'reports' {
            Start-ReportServer
            Open-ProjectUrl "$reportUrl/mvp/index.html"
        }
        'scenes' {
            Start-ReportServer
            Open-ProjectUrl (Get-SelectedReportUrl 'scenes')
        }
        'predictions' {
            Start-ReportServer
            Open-ProjectUrl (Get-SelectedReportUrl 'predictions')
        }
        'annotations' {
            $pythonExe = Get-ProjectPython
            $annotationPort = $localConfiguration.annotation_review.port
            if ($null -eq $annotationPort) { throw 'Configure annotation_review before opening saved annotations.' }
            Write-Host "Saved annotation workspace: http://127.0.0.1:$annotationPort/"
            Write-Host 'Keep this terminal open while the service is running. Ctrl+C stops it.'
            Write-Host 'Reopening does not require repeating completed annotation.'
            & $pythonExe -B 'scripts\review_annotations.py'
            if ($LASTEXITCODE -ne 0) { throw 'Annotation service failed; inspect its terminal message.' }
        }
        'check' {
            $pythonExe = Get-ProjectPython
            & $pythonExe -B 'scripts\check_runtime.py'
            if ($LASTEXITCODE -ne 0) { throw 'Runtime check failed.' }
        }
        'test' {
            $pythonExe = Get-ProjectPython
            & $pythonExe -B -m pytest -q -p no:cacheprovider
            if ($LASTEXITCODE -ne 0) { throw 'Software tests failed.' }
        }
        'stop' { Stop-ReportServer }
        'help' { Write-Host 'mvp.cmd [status|reports|scenes|predictions|annotations|check|test|stop]'; Write-Host 'Run mvp.cmd without an action for the menu.' }
    }
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
} finally {
    Pop-Location
}
