@echo off
setlocal
REM IMPROVED model on the same sizes/data/epochs/scoring as the baseline.
REM   --answer_loss listwise : improvement 1 (drug-only listwise loss)
REM   --drugs_first          : improvement 2 (all candidate drugs get a node)
REM Run from the TraceDR-improved folder (next to TraceDR-model), TraceDR env active:   run_improved.bat
REM Data is read from ..\TraceDR-model\intermediate_data (not copied).
REM Finished sizes are skipped, so it can be resumed. Progress: log_train_<n>.txt / log_test_<n>.txt

set BENCH=DrugRec_mini
set EPOCHS=2
set TAG=improved
set IMPROVE=--answer_loss listwise --drugs_first
set DATA=--intermediate_dir ..\intermediate_data

for %%N in (200 400 800) do (
    if exist output\%TAG%_%%N\test_metrics.json (
        echo == %%N already done, skipping
    ) else (
        echo == [%%N] training ...
        python train_TraceDR.py --benchmark %BENCH%_%%N --model_name %TAG%_%%N --epochs %EPOCHS% --rank_drugs_only %IMPROVE% %DATA% > log_train_%TAG%_%%N.txt 2>&1
        if errorlevel 1 (
            echo Training failed for %%N - see log_train_%TAG%_%%N.txt
            goto :collect
        )
        echo == [%%N] testing ...
        python train_TraceDR.py --eval --benchmark %BENCH%_%%N --model_name %TAG%_%%N --rank_drugs_only %IMPROVE% %DATA% --test_output output\%TAG%_%%N > log_test_%TAG%_%%N.txt 2>&1
        echo == [%%N] done
    )
)

:collect
python collect_results.py %TAG%
