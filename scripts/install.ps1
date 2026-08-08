param(
    [switch] $VoiceNim,
    [switch] $VoiceLocal,
    [switch] $VoiceAll,
    [string] $TorchBackend = "",
    [switch] $DryRun,
    [switch] $Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]] $RemainingArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoArchiveUrl = "https://github.com/King-Jboy/kingjboy-claude-code/archive/refs/heads/main.zip"
# Windows on ARM emulates x64, whose Python package ecosystem has broader wheel support.
$PythonRequest = "cpython-3.14.0-windows-x86_64-none"
$MinUvVersion = "0.11.16"
$ClaudeInstallUrl = "https://claude.ai/install.ps1"
$UvInstallUrl = "https://astral.sh/uv/install.ps1"
# The desktop app renders its window with WebView2. Windows 11 ships the runtime;
# Windows 10 frequently does not, and without it the shortcut this installer
# creates opens a window that cannot draw. The Evergreen Bootstrapper is the
# redistributable Microsoft publishes for exactly this.
$WebView2InstallUrl = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"
$WebView2DownloadPage = "https://developer.microsoft.com/microsoft-edge/webview2/"
$WebView2ClientKey = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
$FccCommands = @(
    # Include retired and unused entry points so updates reject older FCC
    # processes before replacement, even ones this installer no longer sets up.
    "fcc-desktop",
    "fcc-server",
    "fcc-claude",
    "fcc-codex",
    "fcc-pi",
    "fcc-init",
    "free-claude-code"
)

function Show-Usage {
    @"
Usage: install.ps1 [options]

Installs or updates Free Claude Code, then installs or verifies Claude Code.

Options:
  -VoiceNim              Install NVIDIA NIM voice transcription support.
  -VoiceLocal            Install local Whisper voice transcription support.
  -VoiceAll              Install all voice transcription backends.
  -TorchBackend VALUE    Use a uv PyTorch backend, such as cu130. Requires local voice.
  -DryRun                Print commands without running them.
  -Help                  Show this help text.
"@
}

function Write-Step {
    param([string] $Message)

    Write-Host ""
    Write-Host "==> $Message"
}

function Format-Argument {
    param([string] $Value)

    if ($Value -match '^[A-Za-z0-9_./:@%+=,\[\]\\-]+$') {
        return $Value
    }

    return "'" + ($Value -replace "'", "''") + "'"
}

function Format-Command {
    param(
        [string] $FilePath,
        [string[]] $Arguments = @()
    )

    $parts = @($FilePath) + $Arguments
    return ($parts | ForEach-Object { Format-Argument ([string] $_) }) -join " "
}

function Invoke-NativeCommand {
    param(
        [string] $FilePath,
        [string[]] $Arguments = @()
    )

    $commandText = Format-Command -FilePath $FilePath -Arguments $Arguments
    Write-Host "+ $commandText"
    if ($DryRun) {
        return
    }

    $global:LASTEXITCODE = 0
    & $FilePath @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: $commandText"
    }
}

function Invoke-Utf8NativeCapture {
    param(
        [string] $FilePath,
        [string[]] $Arguments = @()
    )

    $commandText = Format-Command -FilePath $FilePath -Arguments $Arguments
    Write-Host "+ $commandText"
    $originalOutputEncoding = [Console]::OutputEncoding
    try {
        [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
        $global:LASTEXITCODE = 0
        $output = & $FilePath @Arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        [Console]::OutputEncoding = $originalOutputEncoding
    }
    if ($exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: $commandText"
    }

    return ($output | Out-String).Trim()
}

function Get-ApplicationCommand {
    param([string] $Name)

    $commands = @(Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue)
    if ($commands.Count -eq 0) {
        return $null
    }

    return $commands[0]
}

function Get-PowerShellExecutable {
    param([string] $PowerShellHome = $PSHOME)

    $executableName = if ($PSVersionTable.PSEdition -eq "Core") {
        "pwsh.exe"
    }
    else {
        "powershell.exe"
    }
    $bundledExecutable = Join-Path $PowerShellHome $executableName
    if (Test-Path -LiteralPath $bundledExecutable -PathType Leaf) {
        return $bundledExecutable
    }

    $pathCommand = Get-ApplicationCommand ([IO.Path]::GetFileNameWithoutExtension($executableName))
    if ($pathCommand) {
        return $pathCommand.Source
    }

    throw "Unable to locate a PowerShell executable for the downloaded installer."
}

function Add-PathEntry {
    param([string] $PathEntry)

    if ([string]::IsNullOrWhiteSpace($PathEntry)) {
        return
    }

    $separator = [IO.Path]::PathSeparator
    $entries = @()
    if (-not [string]::IsNullOrEmpty($env:Path)) {
        $entries = $env:Path -split [regex]::Escape([string] $separator)
    }

    if ($entries -notcontains $PathEntry) {
        $env:Path = "$PathEntry$separator$env:Path"
    }
}

function Add-KnownBinDirectories {
    if (-not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
        Add-PathEntry (Join-Path $env:USERPROFILE ".local\bin")
    }
    if (-not [string]::IsNullOrWhiteSpace($env:APPDATA)) {
        # Claude Code can be installed through npm as well as its own installer.
        Add-PathEntry (Join-Path $env:APPDATA "npm")
    }
}

function Assert-NoFccProcessesRunning {
    $running = @()
    foreach ($commandName in $FccCommands) {
        $processes = @(Get-Process -Name $commandName -ErrorAction SilentlyContinue)
        foreach ($process in $processes) {
            $running += "$commandName (PID $($process.Id))"
        }
    }

    if ($running.Count -gt 0) {
        throw "Free Claude Code is still running ($($running -join ', ')). Stop those processes, then rerun the installer."
    }
}

function Invoke-DownloadedPowerShellInstaller {
    param(
        [string] $Url,
        [string] $Name
    )

    if ($DryRun) {
        Write-Host "+ irm $Url -OutFile <temporary-script>"
        Write-Host "+ powershell -NoProfile -ExecutionPolicy Bypass -File <temporary-script>"
        return
    }

    $temporaryScript = Join-Path ([IO.Path]::GetTempPath()) ("fcc-install-" + [guid]::NewGuid().ToString("N") + ".ps1")
    try {
        Write-Host "+ irm $Url -OutFile $(Format-Argument $temporaryScript)"
        Invoke-RestMethod -Uri $Url -OutFile $temporaryScript -ErrorAction Stop
        if ((-not (Test-Path -LiteralPath $temporaryScript)) -or ((Get-Item -LiteralPath $temporaryScript).Length -eq 0)) {
            throw "The downloaded $Name installer was empty."
        }

        $tokens = $null
        $parseErrors = $null
        [System.Management.Automation.Language.Parser]::ParseFile(
            $temporaryScript,
            [ref] $tokens,
            [ref] $parseErrors
        ) | Out-Null
        if ($parseErrors.Count -gt 0) {
            throw "The downloaded $Name installer from '$Url' is not valid PowerShell. A network proxy or filter may have replaced it with an HTML response."
        }

        $powerShellPath = Get-PowerShellExecutable
        Invoke-NativeCommand -FilePath $powerShellPath -Arguments @(
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            $temporaryScript
        )
    }
    finally {
        Remove-Item -LiteralPath $temporaryScript -Force -ErrorAction SilentlyContinue
    }
}

function Assert-WindowsExecutable {
    param(
        [string] $Path,
        [string] $Url,
        [string] $Name
    )

    if ((-not (Test-Path -LiteralPath $Path)) -or ((Get-Item -LiteralPath $Path).Length -eq 0)) {
        throw "The downloaded $Name installer was empty."
    }

    # The same failure the PowerShell installers catch by parsing: a proxy or
    # captive portal answering with an HTML page instead of the file. Every
    # Windows executable opens with 'MZ'.
    $header = [byte[]]::new(2)
    $stream = [IO.File]::OpenRead($Path)
    try {
        $read = $stream.Read($header, 0, 2)
    }
    finally {
        $stream.Dispose()
    }
    if (($read -lt 2) -or ($header[0] -ne 0x4D) -or ($header[1] -ne 0x5A)) {
        throw "The downloaded $Name installer from '$Url' is not a Windows executable. A network proxy or filter may have replaced it with an HTML response."
    }
}

function Get-WebView2RuntimeVersion {
    # EdgeUpdate's client key is what Microsoft documents as the presence check.
    # A file probe would be guesswork: the runtime lives in a versioned directory
    # whose location has moved between releases. Per-machine installs register
    # under HKLM, and on 64-bit Windows under WOW6432Node; per-user under HKCU.
    $registryPaths = @(
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\$WebView2ClientKey",
        "HKLM:\SOFTWARE\Microsoft\EdgeUpdate\Clients\$WebView2ClientKey",
        "HKCU:\SOFTWARE\Microsoft\EdgeUpdate\Clients\$WebView2ClientKey"
    )

    foreach ($registryPath in $registryPaths) {
        $entry = Get-ItemProperty -Path $registryPath -Name "pv" -ErrorAction SilentlyContinue
        if (-not $entry) {
            continue
        }
        if ($entry.PSObject.Properties.Match("pv").Count -eq 0) {
            continue
        }
        $version = [string] $entry.pv
        # Removing the runtime leaves the key behind holding a 0.0.0.0 stub.
        if ((-not [string]::IsNullOrWhiteSpace($version)) -and ($version -ne "0.0.0.0")) {
            return $version
        }
    }

    return $null
}

function Install-WebView2Runtime {
    $bootstrapper = Join-Path ([IO.Path]::GetTempPath()) ("fcc-webview2-" + [guid]::NewGuid().ToString("N") + ".exe")
    try {
        Write-Host "+ irm $WebView2InstallUrl -OutFile $(Format-Argument $bootstrapper)"
        Invoke-RestMethod -Uri $WebView2InstallUrl -OutFile $bootstrapper -ErrorAction Stop
        Assert-WindowsExecutable -Path $bootstrapper -Url $WebView2InstallUrl -Name "WebView2 runtime"

        $arguments = @("/silent", "/install")
        Write-Host "+ $(Format-Command -FilePath $bootstrapper -Arguments $arguments)"
        $process = Start-Process -FilePath $bootstrapper -ArgumentList $arguments -Wait -PassThru
        try {
            $exitCode = $process.ExitCode
        }
        finally {
            $process.Dispose()
        }
        if ($exitCode -ne 0) {
            throw "The WebView2 runtime installer exited with code ${exitCode}."
        }
    }
    finally {
        Remove-Item -LiteralPath $bootstrapper -Force -ErrorAction SilentlyContinue
    }
}

function Ensure-WebView2Runtime {
    if ($DryRun) {
        Write-Host "+ read the WebView2 runtime version from EdgeUpdate"
        Write-Host "+ irm $WebView2InstallUrl -OutFile <temporary-installer>   (only when missing)"
        Write-Host "+ <temporary-installer> /silent /install                   (only when missing)"
        return
    }

    $version = Get-WebView2RuntimeVersion
    if ($version) {
        Write-Host "WebView2 runtime $version is already present."
        return
    }

    # Never fatal. fcc-server and fcc-claude do not open a window, so a terminal
    # install must not fail because a GUI runtime could not be fetched. The
    # warning has to name the consequence, since nothing else here will.
    try {
        Install-WebView2Runtime
    }
    catch {
        Write-Warning "Could not install the WebView2 runtime: $($_.Exception.Message)"
        Write-Warning "The desktop shortcut needs it. Install it from $WebView2DownloadPage and reopen the shortcut. fcc-server and fcc-claude work without it."
        return
    }

    $version = Get-WebView2RuntimeVersion
    if ($version) {
        Write-Host "WebView2 runtime $version installed."
        return
    }
    Write-Warning "The WebView2 installer finished but EdgeUpdate reports no runtime. The desktop shortcut may not open; fcc-server and fcc-claude are unaffected."
}

function Confirm-Application {
    param(
        [string] $CommandName,
        [string] $DisplayName
    )

    if ($DryRun) {
        Write-Host "+ $CommandName --version"
        return
    }

    $command = Get-ApplicationCommand $CommandName
    if (-not $command) {
        throw "$DisplayName was installed, but '$CommandName' is not available on PATH."
    }
    Invoke-NativeCommand -FilePath $command.Source -Arguments @("--version")
}

function Ensure-ClaudeCode {
    if (Get-ApplicationCommand "claude") {
        Write-Host "Claude Code already found on PATH; verifying it."
    }
    else {
        Invoke-DownloadedPowerShellInstaller -Url $ClaudeInstallUrl -Name "Claude Code"
        Add-KnownBinDirectories
    }

    Confirm-Application -CommandName "claude" -DisplayName "Claude Code"
}

function Convert-UvVersionOutput {
    param([string] $Output)

    if ([string]::IsNullOrWhiteSpace($Output)) {
        return ""
    }

    if ($Output -match '(?m)(?:^|\s)(?:uv\s+)?(?<version>\d+\.\d+\.\d+(?:[-+][0-9A-Za-z][0-9A-Za-z.-]*)?)\b') {
        return $Matches["version"]
    }

    return ""
}

function Get-UvVersion {
    param([string] $UvPath)

    $output = Invoke-Utf8NativeCapture -FilePath $UvPath -Arguments @("--version")
    $version = Convert-UvVersionOutput $output
    if ([string]::IsNullOrWhiteSpace($version)) {
        throw "uv is present, but 'uv --version' did not return a valid version."
    }

    return $version
}

function Test-SupportedUvVersion {
    param(
        [string] $Version,
        [string] $Minimum
    )

    $parsedVersion = Convert-UvVersionOutput $Version
    $parsedMinimum = Convert-UvVersionOutput $Minimum
    if ([string]::IsNullOrWhiteSpace($parsedVersion) -or [string]::IsNullOrWhiteSpace($parsedMinimum)) {
        throw "Unable to compare uv versions."
    }
    if ($parsedVersion.Contains("-")) {
        return $false
    }

    $normalizedVersion = $parsedVersion -replace '\+.*$', ''
    $normalizedMinimum = $parsedMinimum -replace '\+.*$', ''

    return ([version] $normalizedVersion) -ge ([version] $normalizedMinimum)
}

function Confirm-Uv {
    if ($DryRun) {
        Write-Host "+ uv --version"
        return
    }

    $uvCommand = Get-ApplicationCommand "uv"
    if (-not $uvCommand) {
        throw "uv was installed, but it is not available on PATH."
    }

    $version = Get-UvVersion $uvCommand.Source
    if (-not (Test-SupportedUvVersion -Version $version -Minimum $MinUvVersion)) {
        throw "Stable uv $MinUvVersion or newer is required; found uv $version after installation."
    }
    Write-Host "Verified uv $version."
}

function Ensure-Uv {
    if ($DryRun) {
        if (Get-ApplicationCommand "uv") {
            Write-Host "+ uv --version"
            Write-Host "A compatible existing uv will be left unchanged; an obsolete one will be replaced by the standalone installer."
        }
        else {
            Write-Host "uv is not installed; the current standalone uv would be installed."
            Invoke-DownloadedPowerShellInstaller -Url $UvInstallUrl -Name "uv"
            Confirm-Uv
        }
        return
    }

    $uvCommand = Get-ApplicationCommand "uv"
    if ($uvCommand) {
        $version = Get-UvVersion $uvCommand.Source
        if (Test-SupportedUvVersion -Version $version -Minimum $MinUvVersion) {
            Write-Host "uv $version already satisfies >=$MinUvVersion; leaving it unchanged."
            return
        }
        Write-Host "uv $version does not satisfy stable >=$MinUvVersion; installing the current standalone uv."
    }
    else {
        Write-Host "uv is not installed; installing the current standalone uv."
    }

    Invoke-DownloadedPowerShellInstaller -Url $UvInstallUrl -Name "uv"
    Add-KnownBinDirectories
    Confirm-Uv
}

function Get-PackageSpec {
    $includeNim = $VoiceNim
    $includeLocal = $VoiceLocal

    if ($VoiceAll) {
        $includeNim = $true
        $includeLocal = $true
    }

    if ($includeNim -and $includeLocal) {
        return "free-claude-code[voice,voice_local] @ $RepoArchiveUrl"
    }
    if ($includeNim) {
        return "free-claude-code[voice] @ $RepoArchiveUrl"
    }
    if ($includeLocal) {
        return "free-claude-code[voice_local] @ $RepoArchiveUrl"
    }
    return "free-claude-code @ $RepoArchiveUrl"
}

function Install-FreeClaudeCode {
    Assert-NoFccProcessesRunning
    $packageSpec = Get-PackageSpec
    $arguments = @(
        "tool",
        "install",
        "--force",
        "--refresh-package",
        "free-claude-code",
        "--python",
        $PythonRequest
    )
    if (-not [string]::IsNullOrWhiteSpace($TorchBackend)) {
        $arguments += @("--torch-backend", $TorchBackend)
    }
    $arguments += $packageSpec

    $uvPath = "uv"
    if (-not $DryRun) {
        $uvCommand = Get-ApplicationCommand "uv"
        if (-not $uvCommand) {
            throw "uv is not available for the Free Claude Code installation."
        }
        $uvPath = $uvCommand.Source
    }
    Invoke-NativeCommand -FilePath $uvPath -Arguments $arguments
}

function Export-FccDesktopIcon {
    param(
        [string] $DesktopCommand,
        [string] $IconPath
    )

    $arguments = @("--export-icon", $IconPath)
    $commandText = Format-Command -FilePath $DesktopCommand -Arguments $arguments
    Write-Host "+ $commandText"
    if ($DryRun) {
        return
    }

    # PowerShell does not wait when directly invoking a Windows GUI executable.
    $process = Start-Process `
        -FilePath $DesktopCommand `
        -ArgumentList @("--export-icon", ('"' + $IconPath + '"')) `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    try {
        $exitCode = $process.ExitCode
    }
    finally {
        $process.Dispose()
    }
    if ($exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: $commandText"
    }
    if (-not (Test-Path -LiteralPath $IconPath -PathType Leaf)) {
        throw "Free Claude Code did not export its Windows app icon to '$IconPath'."
    }
}

function Configure-AndConfirmFreeClaudeCode {
    $iconPath = Join-Path $env:USERPROFILE ".fcc\app-icon.ico"
    if ($DryRun) {
        Write-Host "+ uv tool update-shell"
        Write-Host "+ uv tool dir --bin"
        Write-Host "+ verify fcc-desktop, fcc-server, and fcc-claude in the uv tool bin directory"
        Write-Host "+ fcc-server --version"
        Export-FccDesktopIcon `
            -DesktopCommand "<uv-tool-bin>\fcc-desktop.exe" `
            -IconPath $iconPath
        Install-FccDesktopShortcuts `
            -DesktopCommand "<uv-tool-bin>\fcc-desktop.exe" `
            -IconPath $iconPath
        return
    }

    $uvCommand = Get-ApplicationCommand "uv"
    if (-not $uvCommand) {
        throw "uv is not available for PATH configuration."
    }
    Invoke-NativeCommand -FilePath $uvCommand.Source -Arguments @("tool", "update-shell")
    $toolBin = Invoke-Utf8NativeCapture -FilePath $uvCommand.Source -Arguments @("tool", "dir", "--bin")
    if ([string]::IsNullOrWhiteSpace($toolBin)) {
        throw "uv returned an empty tool bin directory."
    }

    Add-PathEntry $toolBin
    $toolBinPath = ([IO.Path]::GetFullPath($toolBin)).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    $installedCommands = @{}
    foreach ($commandName in @("fcc-desktop", "fcc-server", "fcc-claude")) {
        $command = Get-ApplicationCommand $commandName
        if (-not $command) {
            throw "Free Claude Code installation did not create '$commandName'."
        }
        $commandDirectory = ([IO.Path]::GetFullPath((Split-Path -Parent $command.Source))).TrimEnd(
            [IO.Path]::DirectorySeparatorChar,
            [IO.Path]::AltDirectorySeparatorChar
        )
        if (-not $commandDirectory.Equals($toolBinPath, [StringComparison]::OrdinalIgnoreCase)) {
            throw "'$commandName' resolved outside the uv tool bin directory: $($command.Source)"
        }
        $installedCommands[$commandName] = $command.Source
    }

    Invoke-NativeCommand -FilePath $installedCommands["fcc-server"] -Arguments @("--version")
    Export-FccDesktopIcon `
        -DesktopCommand $installedCommands["fcc-desktop"] `
        -IconPath $iconPath
    Install-FccDesktopShortcuts `
        -DesktopCommand $installedCommands["fcc-desktop"] `
        -IconPath $iconPath
}

function Test-EquivalentPath {
    param(
        [string] $Left,
        [string] $Right
    )

    if ([string]::IsNullOrWhiteSpace($Left) -or [string]::IsNullOrWhiteSpace($Right)) {
        return $false
    }
    try {
        return [string]::Equals(
            [IO.Path]::GetFullPath($Left),
            [IO.Path]::GetFullPath($Right),
            [StringComparison]::OrdinalIgnoreCase
        )
    }
    catch {
        return $false
    }
}

function Install-FccDesktopShortcuts {
    param(
        [string] $DesktopCommand,
        [string] $IconPath
    )

    $shortcutPaths = @(
        (Join-Path $env:USERPROFILE "Desktop\Free Claude Code.lnk"),
        (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Free Claude Code.lnk")
    )
    foreach ($shortcutPath in $shortcutPaths) {
        Write-Host "+ create shortcut $(Format-Argument $shortcutPath) -> $(Format-Argument $DesktopCommand)"
    }
    if ($DryRun) {
        return
    }

    $shell = New-Object -ComObject WScript.Shell
    foreach ($shortcutPath in $shortcutPaths) {
        if (Test-Path -LiteralPath $shortcutPath) {
            try {
                $existingShortcut = $shell.CreateShortcut($shortcutPath)
                $isFccShortcut = Test-EquivalentPath -Left $existingShortcut.TargetPath -Right $DesktopCommand
            }
            catch {
                $isFccShortcut = $false
            }
            if (-not $isFccShortcut) {
                Write-Host "A shortcut not managed by Free Claude Code already exists at $shortcutPath; leaving it unchanged."
                continue
            }
        }
        $parent = Split-Path -Parent $shortcutPath
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
        $shortcut = $shell.CreateShortcut($shortcutPath)
        $shortcut.TargetPath = $DesktopCommand
        $shortcut.WorkingDirectory = $env:USERPROFILE
        $shortcut.IconLocation = "$IconPath,0"
        $shortcut.Description = "Run Free Claude Code in the background"
        $shortcut.Save()
    }
}

if ($Help) {
    Show-Usage
    return
}

if ($RemainingArgs.Count -gt 0) {
    Show-Usage
    throw "Unknown option: $($RemainingArgs -join ' ')"
}

if ((-not [string]::IsNullOrWhiteSpace($TorchBackend)) -and (-not ($VoiceLocal -or $VoiceAll))) {
    throw "-TorchBackend requires -VoiceLocal or -VoiceAll."
}

Add-KnownBinDirectories

Write-Step "Checking for running Free Claude Code processes"
Assert-NoFccProcessesRunning

Write-Step "Ensuring Claude Code is installed"
Ensure-ClaudeCode

Write-Step "Ensuring uv $MinUvVersion or newer is installed"
Ensure-Uv

Write-Step "Installing or updating Free Claude Code"
Install-FreeClaudeCode

Write-Step "Ensuring the desktop app can render its window"
Ensure-WebView2Runtime

Write-Step "Configuring PATH and verifying Free Claude Code"
Configure-AndConfirmFreeClaudeCode

Write-Host ""
if ($DryRun) {
    Write-Host "Dry run complete. No changes were made."
}
else {
    Write-Host "Free Claude Code is installed and verified. Open the Free Claude Code desktop shortcut to run it in the background."
    Write-Host "For terminal use, start the proxy with: fcc-server"
    Write-Host "Run Claude Code with: fcc-claude"
}
