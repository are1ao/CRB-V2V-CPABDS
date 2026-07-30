@echo off
echo ================================================================================
echo   FINAL FIX - 清除缓存并重新运行
echo ================================================================================
echo.
echo [Found Issue]
echo   Attack vehicle 147 initial reputation: 0.670 (should be 1.0)
echo   Normal vehicle 146 initial reputation: 1.000 (correct)
echo.
echo [Root Cause]
echo   _get_meta() did not set trust_vector.direct and indirect
echo   They got overwritten by consistency_ratio in first update
echo.
echo [Fix Applied]
echo   improved_reputation_engine.py _get_meta() method
echo   Now explicitly sets all three trust_vector fields to 1.0
echo.
echo ================================================================================
echo.

echo [Step 1] Clearing Python cache...
if exist __pycache__ (
    rmdir /s /q __pycache__
    echo   Cache cleared
) else (
    echo   No cache found
)

echo.
echo [Step 2] Ready to rerun experiment
echo.

choice /C YN /M "Rerun experiment now (1 episode per scenario, ~10 min)"

if errorlevel 2 goto :skip
if errorlevel 1 goto :run

:run
echo.
echo ================================================================================
echo   Running Experiment (Final Fix)
echo ================================================================================
echo.

set EPISODES=1
set SCENARIOS=teleport,drift,reverse,brake,obstacle
python run_complete_experiment.py

if errorlevel 1 (
    echo.
    echo [ERROR] Experiment failed
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo   Generating Visualizations
echo ================================================================================
echo.

python advanced_visualization.py

echo.
echo ================================================================================
echo   Verifying Results
echo ================================================================================
echo.

python check_latest_results.py

echo.
echo ================================================================================
echo   Success!
echo ================================================================================
echo.
echo Check visualizations:
echo   visualizations\comparison_obstacle.png
echo.
echo Expected:
echo   [OK] Attack vehicle 147 initial = 1.0
echo   [OK] Detection Delay subplot visible
echo   [OK] Reputation Separation subplot visible
echo.
goto :end

:skip
echo.
echo Skipped. To run manually:
echo   1. Delete __pycache__ folder
echo   2. set EPISODES=1
echo   3. set SCENARIOS=teleport,drift,reverse,brake,obstacle
echo   4. python run_complete_experiment.py
echo   5. python advanced_visualization.py
echo.

:end
pause
