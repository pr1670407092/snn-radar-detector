@echo off
REM ============================================================
REM  SNN radar detector - DaVinci Pro board deploy script
REM
REM    (no arg)  build + program  (~60 min)
REM    program   program existing bitstream only  (~30 s)
REM    verify    read back DONE bit
REM    probe     list JTAG devices
REM
REM  NOTE: this file is intentionally pure ASCII. Chinese comments in
REM  a UTF-8 .bat get mangled by cmd.exe on a GBK code page and break
REM  parsing. Chinese docs live in README.md and 00_HOWTO (root of pkg).
REM
REM  Vivado location: auto-detected below, or set VIVADO_BIN first.
REM ============================================================
setlocal enabledelayedexpansion
set PROCESSOR_ARCHITECTURE=AMD64
set HERE=%~dp0

REM ---------- locate Vivado ----------
set VIVADO=
if defined VIVADO_BIN if exist "%VIVADO_BIN%\vivado.bat" set VIVADO=%VIVADO_BIN%\vivado.bat
if not defined VIVADO for %%P in (
    "F:\vivado\2025.2\Vivado\bin\vivado.bat"
    "G:\app\vivado2025.2\2025.2\Vivado\bin\vivado.bat"
    "C:\Xilinx\Vivado\2025.2\bin\vivado.bat"
    "C:\Xilinx\Vivado\2024.2\bin\vivado.bat"
    "C:\Xilinx\Vivado\2023.2\bin\vivado.bat"
    "C:\Xilinx\Vivado\2022.2\bin\vivado.bat"
    "D:\Xilinx\Vivado\2025.2\bin\vivado.bat"
    "E:\Xilinx\Vivado\2025.2\bin\vivado.bat"
) do (
    if not defined VIVADO if exist %%P set VIVADO=%%~P
)
if not defined VIVADO (
    echo *** Vivado not found ***
    echo Set VIVADO_BIN to the Vivado bin directory, e.g.
    echo     set VIVADO_BIN=C:\Xilinx\Vivado\2025.2\bin
    echo then run this script again.
    pause
    exit /b 1
)
echo Using Vivado: %VIVADO%

if "%1"=="program" goto :program
if "%1"=="probe"   goto :probe
if "%1"=="verify"  goto :verify

REM ---------------------------------------------------------------
REM  Build stage: put TEMP on this package's drive. If the system
REM  drive is nearly full, Vivado hangs silently during place/route
REM  (no error, no exit, CPU idle).
REM ---------------------------------------------------------------
set TEMP=%~d0\vivado_tmp
set TMP=%~d0\vivado_tmp
if not exist "%TEMP%" mkdir "%TEMP%"

echo.
echo [1/2] synthesis + implementation + bitstream  (~60 min)
echo       temp dir: %TEMP%
call "%VIVADO%" -mode batch -notrace -source "%HERE%scripts\build_board.tcl" -log "%HERE%build_board.log" -journal "%HERE%build_board.jou"
if errorlevel 1 goto :fail

:program
REM Program/verify must use the DEFAULT temp dir: pointing TEMP to
REM another drive makes connect_hw_server fail (Labtools 27-3733).
set TEMP=
set TMP=
echo.
echo [2/2] JTAG program
call "%VIVADO%" -mode batch -notrace -source "%HERE%scripts\program_board.tcl" -log "%HERE%program_board.log" -journal "%HERE%program_board.jou"
if errorlevel 1 goto :fail
echo.
echo Done. Press RESET on the board to re-run the 8 test cases.
echo   LED1 (Y7) on  = all 8 cases finished
echo   LED3 (V9) on  = 8/8 PASS
goto :done

:verify
call "%VIVADO%" -mode batch -notrace -source "%HERE%scripts\verify_board.tcl" -log "%HERE%verify_board.log"
if errorlevel 1 goto :fail
echo Look for BIT5_DONE in verify_board.log: 1 means configured OK.
goto :done

:probe
call "%VIVADO%" -mode batch -notrace -source "%HERE%scripts\probe_jtag.tcl" -log "%HERE%probe_jtag.log"
if errorlevel 1 goto :fail
echo Look for PROBE_DEVICE lines in probe_jtag.log.
goto :done

:fail
echo.
echo *** FAILED - see the matching .log file ***
echo     If the build hangs: make sure the package drive has GBs free.
pause
exit /b 1

:done
pause
exit /b 0
