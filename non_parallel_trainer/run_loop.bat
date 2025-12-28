@echo off
REM -------------------------------------------------------------
REM  Non-parallel self-play + training alternating loop
REM  Usage: run_loop.bat [LOOP_COUNT]
REM  LOOP_COUNT empty -> 1000 iterations; use -1 for infinite.
REM -------------------------------------------------------------
SETLOCAL ENABLEDELAYEDEXPANSION

REM Resolve repo root
SET "SCRIPT_DIR=%~dp0"
FOR %%I IN ("%SCRIPT_DIR%..") DO SET REPO_ROOT=%%~fI
PUSHD "%REPO_ROOT%"

REM Defaults (override via environment before calling)
IF NOT DEFINED EPISODES_PER_GEN SET EPISODES_PER_GEN=300
IF NOT DEFINED WORKERS SET WORKERS=16
IF NOT DEFINED TRAIN_UPDATES SET TRAIN_UPDATES=200
IF NOT DEFINED MAX_FILES_PER_TRAIN SET MAX_FILES_PER_TRAIN=100
REM data/ の保存上限（FIFO）。100を既定に設定
IF NOT DEFINED DATA_MAX_FILES SET DATA_MAX_FILES=100
IF NOT DEFINED DATA_DIR SET DATA_DIR=data
IF NOT DEFINED LOG_DIR SET LOG_DIR=logs
IF NOT DEFINED CKPT_DIR SET CKPT_DIR=checkpoints
IF NOT DEFINED VERSION_INTERVAL SET VERSION_INTERVAL=10000
IF NOT DEFINED VERBOSE_PHASE_LOG SET VERBOSE_PHASE_LOG=1

REM Prefer venv python
SET "VENV_PY=%REPO_ROOT%\.venv\Scripts\python.exe"
IF EXIST "%VENV_PY%" (
  SET "PYTHON_EXE=%VENV_PY%"
) ELSE IF NOT DEFINED PYTHON_EXE (
  SET "PYTHON_EXE=python"
)

REM Loop count
SET LOOP_COUNT=%1
IF "%LOOP_COUNT%"=="" SET LOOP_COUNT=5000

ECHO [RUN-LOOP] Start episodes/gen=%EPISODES_PER_GEN% workers=%WORKERS% updates/train=%TRAIN_UPDATES% loop_count=%LOOP_COUNT%
ECHO [RUN-LOOP] Data=%DATA_DIR% Log=%LOG_DIR% Checkpoints=%CKPT_DIR% Python=%PYTHON_EXE%
ECHO [RUN-LOOP] version_interval(episodes)=%VERSION_INTERVAL%
ECHO [RUN-LOOP] Press Ctrl+C to stop.

SET ITER=0

:MAIN_LOOP
IF NOT "%LOOP_COUNT%"=="-1" (
  IF %ITER% GEQ %LOOP_COUNT% GOTO END
)
SET /A ITER=%ITER%+1
ECHO.
ECHO ===================== Iteration %ITER% =====================

REM ---- Adjust episodes/updates based on file count and episode count ----
REM Phase 1: Until 30 files accumulated -> self-play only (no training)
REM Phase 2: 30+ files, episodes < 100,000 -> updates=50
REM Phase 3: 30+ files, episodes >= 100,000 -> updates=90

REM Count selfplay files in data directory
SET FILE_COUNT=0
FOR %%F IN ("!DATA_DIR!\selfplay_ep*.joblib") DO SET /A FILE_COUNT+=1

REM Get cumulative episodes from meta.json (fallback to iteration * episodes_per_gen)
SET CUMULATIVE_EPISODES=0
IF EXIST "!DATA_DIR!\meta.json" (
  FOR /F "tokens=2 delims=:," %%A IN ('FINDSTR /C:"episodes_cumulative" "!DATA_DIR!\meta.json"') DO (
    SET "CUMULATIVE_EPISODES=%%A"
    SET "CUMULATIVE_EPISODES=!CUMULATIVE_EPISODES: =!"
  )
)
IF "!CUMULATIVE_EPISODES!"=="" SET CUMULATIVE_EPISODES=0

REM Determine training updates based on conditions
SET "CURRENT_EPISODES=!EPISODES_PER_GEN!"

IF !FILE_COUNT! LSS 30 (
  REM Phase 1: Accumulating files - no training
  SET "CURRENT_UPDATES=0"
  ECHO [PHASE] Warm-up: files=!FILE_COUNT!/30 episodes=!CUMULATIVE_EPISODES! training=SKIP
) ELSE IF !CUMULATIVE_EPISODES! LSS 100000 (
  REM Phase 2: 30+ files but less than 100k episodes - updates=50
  SET "CURRENT_UPDATES=50"
  ECHO [PHASE] Early training: files=!FILE_COUNT! episodes=!CUMULATIVE_EPISODES! updates=50
) ELSE (
  REM Phase 3: 30+ files and 100k+ episodes - updates=90
  SET "CURRENT_UPDATES=90"
  ECHO [PHASE] Full training: files=!FILE_COUNT! episodes=!CUMULATIVE_EPISODES! updates=90
)

REM ---- Phase 1: Self-play data generation ----
ECHO [PHASE self-play] episodes=!CURRENT_EPISODES! workers=%WORKERS%
"%PYTHON_EXE%" -m non_parallel_trainer.non_parallel_self_play --episodes !CURRENT_EPISODES! --workers !WORKERS! --data-dir "!DATA_DIR!" --log-dir "!LOG_DIR!" --checkpoint-dir "!CKPT_DIR!"
SET SP_EXIT=%ERRORLEVEL%
ECHO [DEBUG] self-play exit code=%SP_EXIT%
IF %SP_EXIT% NEQ 0 (
  ECHO [ERROR] self-play returned non-zero exit code %SP_EXIT% iteration=!ITER!
  GOTO ERROR_PAUSE
)
REM 生成後に古い自己対局ファイルをFIFOで整理（上限 DATA_MAX_FILES）
ECHO [PHASE prune-data] max-files=%DATA_MAX_FILES%
"%PYTHON_EXE%" -m non_parallel_trainer.prune_data --data-dir "!DATA_DIR!" --max-files !DATA_MAX_FILES!
IF %VERBOSE_PHASE_LOG%==1 ECHO [TIME] self-play finished at %DATE% %TIME%

REM ---- Phase 2: Training ingest + updates ----
REM Skip training if CURRENT_UPDATES is 0
IF NOT DEFINED CURRENT_UPDATES SET "CURRENT_UPDATES=0"
IF "!CURRENT_UPDATES!"=="0" (
  ECHO [PHASE train] SKIPPED (warm-up phase)
) ELSE (
  ECHO [PHASE train] updates=!CURRENT_UPDATES! max-files=%MAX_FILES_PER_TRAIN%
  "%PYTHON_EXE%" -m non_parallel_trainer.non_parallel_trainer --data-dir "!DATA_DIR!" --log-dir "!LOG_DIR!" --checkpoint-dir "!CKPT_DIR!" --updates !CURRENT_UPDATES! --max-files !MAX_FILES_PER_TRAIN! --version-interval !VERSION_INTERVAL!
  SET TR_EXIT=%ERRORLEVEL%
  ECHO [DEBUG] trainer exit code=%TR_EXIT%
  IF !TR_EXIT! NEQ 0 (
    ECHO [ERROR] training returned non-zero exit code !TR_EXIT! iteration=!ITER!
    GOTO ERROR_PAUSE
  )
  IF %VERBOSE_PHASE_LOG%==1 ECHO [TIME] training finished at %DATE% %TIME%
)

GOTO MAIN_LOOP

:ERROR_PAUSE
ECHO.
ECHO Press any key to continue loop, or Ctrl+C to abort.
PAUSE >NUL
GOTO MAIN_LOOP

:END
ECHO.
ECHO [RUN-LOOP] Completed %ITER% iteration(s). Exiting.
POPD
ENDLOCAL
EXIT /B 0
