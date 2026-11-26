@echo off
setlocal
REM IMPROVEMENT 3 - safety-aware ranking. Run from TraceDR-model\safety (TraceDR env):   run_safety.bat
REM All results on the 1000-patient test set (DrugRec_test1000), same as the ablation re-scoring.
REM   A) re-rank the EXISTING improved model at test time (no training):  beta = 2 (soft) and 100 (hard)
REM   B) train improved + safety penalty (safety_weight 1.0) on the same 800 patients / 2 epochs, then test
REM Needs: ..\intermediate_data\DrugRec_mini_800 and DrugRec_test1000, ..\improved_model\saved\improved_800

set DATA=--intermediate_dir ..\intermediate_data
set IMP=--rank_drugs_only --answer_loss listwise --drugs_first
set T=DrugRec_test1000
set CKPT=..\improved_model\saved\improved_800\best_model.pt

echo == [1/4] improved + re-rank beta=2
if not exist output\improved_rr2_t1000_800\test_metrics.json python train_TraceDR.py --eval --benchmark %T% %DATA% %IMP% --safety_rerank 2 --resume_path %CKPT% --test_output output\improved_rr2_t1000_800 > log_rr2.txt 2>&1

echo == [2/4] improved + re-rank beta=100 (hard)
if not exist output\improved_rr100_t1000_800\test_metrics.json python train_TraceDR.py --eval --benchmark %T% %DATA% %IMP% --safety_rerank 100 --resume_path %CKPT% --test_output output\improved_rr100_t1000_800 > log_rr100.txt 2>&1

echo == [3/4] training improved + safety penalty (about 30 min)
if not exist saved\safe_800\best_model.pt (
    python train_TraceDR.py --benchmark DrugRec_mini_800 --model_name safe_800 --epochs 2 %DATA% %IMP% --safety_weight 1.0 > log_train_safe_800.txt 2>&1
    if errorlevel 1 ( echo Training failed - see log_train_safe_800.txt & goto :eof )
)
echo == [4/4] testing improved + safety penalty
if not exist output\safe_t1000_800\test_metrics.json python train_TraceDR.py --eval --benchmark %T% %DATA% %IMP% --safety_weight 1.0 --model_name safe_800 --test_output output\safe_t1000_800 > log_test_safe.txt 2>&1

echo.
findstr /c:"Safety flags" log_rr2.txt
set PT=python paired_test.py --sizes 800 --baseline_root ..\improved_model --baseline_tag improved_t1000 --improved_root .
echo ########## re-rank beta=2  vs IMPROVED ##########
%PT% --improved_tag improved_rr2_t1000
echo ########## re-rank beta=100 (hard)  vs IMPROVED ##########
%PT% --improved_tag improved_rr100_t1000
echo ########## safety penalty  vs IMPROVED ##########
%PT% --improved_tag safe_t1000
echo ########## re-rank beta=100 (hard)  vs BASELINE ##########
python paired_test.py --sizes 800 --baseline_root .. --baseline_tag baseline_t1000 --improved_root . --improved_tag improved_rr100_t1000
echo ########## safety penalty  vs BASELINE ##########
python paired_test.py --sizes 800 --baseline_root .. --baseline_tag baseline_t1000 --improved_root . --improved_tag safe_t1000
