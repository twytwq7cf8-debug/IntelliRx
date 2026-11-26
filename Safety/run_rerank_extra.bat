@echo off
setlocal
REM Two extra scoring runs (no training, ~6-7 min each) on the 1,000-patient test set:
REM   A) drugs-first-only model  + hard safety re-ranking (beta 100)
REM   B) original TraceDR baseline + hard safety re-ranking (beta 100)
REM Run from TraceDR-model\safety (TraceDR env):   run_rerank_extra.bat

set DATA=--intermediate_dir ..\intermediate_data
set T=DrugRec_test1000

echo == [1/2] drugs-first only + re-rank beta=100
if not exist output\drugsfirst_rr100_t1000_800\test_metrics.json python train_TraceDR.py --eval --benchmark %T% %DATA% --rank_drugs_only --drugs_first --safety_rerank 100 --resume_path ..\ablation\saved\abl_drugsfirst_800\best_model.pt --test_output output\drugsfirst_rr100_t1000_800 > log_df_rr100.txt 2>&1

echo == [2/2] baseline + re-rank beta=100
if not exist output\baseline_rr100_t1000_800\test_metrics.json python train_TraceDR.py --eval --benchmark %T% %DATA% --rank_drugs_only --safety_rerank 100 --resume_path ..\saved\baseline_800\best_model.pt --test_output output\baseline_rr100_t1000_800 > log_base_rr100.txt 2>&1

echo.
echo ########## drugs-first + re-rank  vs BASELINE ##########
python paired_test.py --sizes 800 --baseline_root .. --baseline_tag baseline_t1000 --improved_root . --improved_tag drugsfirst_rr100_t1000
echo ########## drugs-first + re-rank  vs  improved + re-rank (final) ##########
python paired_test.py --sizes 800 --baseline_root . --baseline_tag improved_rr100_t1000 --improved_root . --improved_tag drugsfirst_rr100_t1000
echo ########## drugs-first + re-rank  vs  drugs-first only ##########
python paired_test.py --sizes 800 --baseline_root ..\ablation --baseline_tag abl_drugsfirst_t1000 --improved_root . --improved_tag drugsfirst_rr100_t1000
echo ########## baseline + re-rank  vs BASELINE ##########
python paired_test.py --sizes 800 --baseline_root .. --baseline_tag baseline_t1000 --improved_root . --improved_tag baseline_rr100_t1000
