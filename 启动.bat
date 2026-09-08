@echo off
cd /d %~dp0
call .venv\Scripts\activate.bat
set HTTP_PROXY=http://127.0.0.1:7897
set HTTPS_PROXY=http://127.0.0.1:7897
echo 已进入 lrac-lab 环境，直接敲命令吧
cmd /k