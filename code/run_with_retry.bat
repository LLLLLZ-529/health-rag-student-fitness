@echo off
chcp 65001 >nul
REM 自动切换到脚本所在目录，避免硬编码绝对路径
cd /d "%~dp0"

:loop
echo ========================================
echo [%date% %time%] 启动训练...
echo ========================================
python -u 02_run_sft_7b.py

if %errorlevel% equ 0 (
    echo [%date% %time%] 训练正常完成！
    goto :end
) else (
    echo [%date% %time%] 训练异常退出 (exit code=%errorlevel%)，10秒后自动重启...
    timeout /t 10 /nobreak >nul
    goto :loop
)

:end
pause
