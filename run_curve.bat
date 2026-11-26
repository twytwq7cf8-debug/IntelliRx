@echo off
setlocal
REM Baseline learning curve (original TraceDR), sized to finish in under 1 hour on an RTX 4060.
REM Run from TraceDR-model in the Anaconda Prompt with the TraceDR env active:   run_curve.bat
REM Finished sizes are skipped, so it can be resumed. Progress: log_train_<n>.txt / log_test_<n>.txt

set BENCH=DrugRec_mini
set EPOCHS=2
set TAG=baseline

for %%N in (200 400 800) do (
    if exist output\%TAG%_%%N\test_metrics.json (
        echo == %%N already done, skipping
    ) else (
        echo == [%%N] training ...
        python train_TraceDR.py --benchmark %BENCH%_%%N --model_name %TAG%_%%N --epochs %EPOCHS% --rank_drugs_only > log_train_%TAG%_%%N.txt 2>&1
        if errorlevel 1 (
            echo Training failed for %%N - see log_train_%TAG%_%%N.txt
            goto :collect
        )
        echo == [%%N] testing ...
        python train_TraceDR.py --eval --benchmark %BENCH%_%%N --model_name %TAG%_%%N --rank_drugs_only --test_output output\%TAG%_%%N > log_test_%TAG%_%%N.txt 2>&1
        echo == [%%N] done
    )
)

:collect
python collect_results.py %TAG%
