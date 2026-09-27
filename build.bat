@echo off
setlocal enabledelayedexpansion

echo ----------------------------------------------------
echo Starting React/Vite site build...
echo ----------------------------------------------------

cd eda-website
call npm run build
cd ..

if %ERRORLEVEL% equ 0 (
    echo.
    echo [SUCCESS] Site generated in docs/ folder!
    echo Opening browser...
    start docs\index.html
) else (
    echo.
    echo [ERROR] Something went wrong during build.
    echo Make sure you have run 'npm install' in the eda-website directory.
)

pause
