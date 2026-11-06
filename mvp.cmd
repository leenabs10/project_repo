@echo off
rem Run the project menu without changing the system execution policy.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\mvp.ps1" %*
exit /b %errorlevel%
