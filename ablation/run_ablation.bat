@echo off
setlocal
REM ABLATION: which of the two changes gives the gain?
REM   abl_listwise   = original TraceDR + listwise drug-only loss only
REM   abl_drugsfirst = original TraceDR + drugs-first graph only
REM Same data / epochs / scoring as baseline and improved. Run from TraceDR-model\ablation:  run_ablation.bat
REM Default: size 800 only (~50 min for both). For all sizes use:  set SIZES=200 400 800

set BENCH=DrugRec_mini
set EPOCHS=2
set SIZES=800
set DATA=--intermediate_dir ..\intermediate_data

call :variant abl_listwise "--answer_loss listwise"
call :variant abl_drugsfirst "--drugs_first"
python ablation_summary.py
goto :eof

:variant
set TAG=%~1
set FLAGS=%~2
for %%N in (%SIZES%) do (
    if exist output\%TAG%_%%N\test_metrics.json (
        echo == %TAG% %%N already done, skipping
    ) else (
        echo == [%TAG% %%N] training ...
        python train_TraceDR.py --benchmark %BENCH%_%%N --model_name %TAG%_%%N --epochs %EPOCHS% --rank_drugs_only %FLAGS% %DATA% > log_train_%TAG%_%%N.txt 2>&1
        if errorlevel 1 (
            echo Training failed for %TAG% %%N - see log_train_%TAG%_%%N.txt
            goto :eof
        )
        echo == [%TAG% %%N] testing ...
        python train_TraceDR.py --eval --benchmark %BENCH%_%%N --model_name %TAG%_%%N --rank_drugs_only %FLAGS% %DATA% --test_output output\%TAG%_%%N > log_test_%TAG%_%%N.txt 2>&1
        echo == [%TAG% %%N] done
    )
)
goto :eof
