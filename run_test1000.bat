@echo off
setlocal
REM Re-score the four trained 800-patient models on a LARGER held-out test set (1000 patients).
REM No retraining. Run from TraceDR-model (Anaconda Prompt, TraceDR env):   run_test1000.bat
REM Step 1 builds intermediate_data\DrugRec_test1000\test.pkl from the ORIGINAL test split
REM (patients never seen in training). Each model is scored with the same graph settings it was trained with.

set T=DrugRec_test1000
set ROOT=%CD%

if not exist intermediate_data\%T%\test.pkl (
    echo == building 1000-patient test set ...
    python make_subset.py --from_intermediate intermediate_data\DrugRec0716 --out_bench %T% --n_train 1 --n_dev 1 --n_test 1000 --seed 2026
    if errorlevel 1 ( echo make_subset failed & goto :eof )
)

echo == [1/4] baseline
python train_TraceDR.py --eval --benchmark %T% --model_name baseline_800 --rank_drugs_only --test_output output\baseline_t1000_800 > log_t1000_baseline.txt 2>&1

cd /d "%ROOT%\improved_model"
echo == [2/4] improved (both)
python train_TraceDR.py --eval --benchmark %T% --model_name improved_800 --rank_drugs_only --answer_loss listwise --drugs_first --intermediate_dir ..\intermediate_data --test_output output\improved_t1000_800 > log_t1000_improved.txt 2>&1

cd /d "%ROOT%\ablation"
echo == [3/4] listwise only
python train_TraceDR.py --eval --benchmark %T% --model_name abl_listwise_800 --rank_drugs_only --answer_loss listwise --intermediate_dir ..\intermediate_data --test_output output\abl_listwise_t1000_800 > log_t1000_listwise.txt 2>&1
echo == [4/4] drugs-first only
python train_TraceDR.py --eval --benchmark %T% --model_name abl_drugsfirst_800 --rank_drugs_only --drugs_first --intermediate_dir ..\intermediate_data --test_output output\abl_drugsfirst_t1000_800 > log_t1000_drugsfirst.txt 2>&1

cd /d "%ROOT%"
set PT=python improved_model\paired_test.py --sizes 800 --improved_root improved_model --improved_tag improved_t1000
echo.
echo ########## improved vs BASELINE ##########
%PT% --baseline_root . --baseline_tag baseline_t1000
echo ########## improved vs LISTWISE ONLY ##########
%PT% --baseline_root ablation --baseline_tag abl_listwise_t1000
echo ########## improved vs DRUGS-FIRST ONLY ##########
%PT% --baseline_root ablation --baseline_tag abl_drugsfirst_t1000
echo ########## listwise only vs BASELINE ##########
python improved_model\paired_test.py --sizes 800 --baseline_root . --baseline_tag baseline_t1000 --improved_root ablation --improved_tag abl_listwise_t1000
echo ########## drugs-first only vs BASELINE ##########
python improved_model\paired_test.py --sizes 800 --baseline_root . --baseline_tag baseline_t1000 --improved_root ablation --improved_tag abl_drugsfirst_t1000
