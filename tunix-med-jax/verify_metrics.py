# Data from notebooks
baseline_raw_sim_mean = 0.516
baseline_raw_sim_min = 0.015
baseline_raw_sim_max = 0.891
baseline_keyword_f1 = 0.102
baseline_ai_judge = 0.571

finetuned_raw_sim_mean = 0.529
finetuned_keyword_f1 = 0.221
finetuned_ai_judge = 0.590

# Fixed calibration range (using baseline as reference)
rng = baseline_raw_sim_max - baseline_raw_sim_min


def calibrate(raw):
    return (raw - baseline_raw_sim_min) / rng


baseline_semantic_fixed = calibrate(baseline_raw_sim_mean)
finetuned_semantic_fixed = calibrate(finetuned_raw_sim_mean)


def final_score(k, s, a):
    return k * 0.2 + s * 0.4 + a * 0.4


baseline_final = final_score(
    baseline_keyword_f1, baseline_semantic_fixed, baseline_ai_judge
)
finetuned_final = final_score(
    finetuned_keyword_f1, finetuned_semantic_fixed, finetuned_ai_judge
)

print("--- Comparison with FIXED Calibration ---")
print(f"Baseline Semantic Score (Fixed): {baseline_semantic_fixed:.3f}")
print(f"Fine-Tuned Semantic Score (Fixed): {finetuned_semantic_fixed:.3f}")
print(f"Baseline Final Score: {baseline_final:.3f}")
print(f"Fine-Tuned Final Score: {finetuned_final:.3f}")
print(f"Absolute Improvement: {finetuned_final - baseline_final:.3f}")
print(f"Relative Improvement: {(finetuned_final / baseline_final - 1) * 100:.1f}%")

print("\n--- Why the original notebook was misleading ---")
finetuned_local_sim_min = 0.048
finetuned_local_sim_max = 1.000
finetuned_semantic_local = (finetuned_raw_sim_mean - finetuned_local_sim_min) / (
    finetuned_local_sim_max - finetuned_local_sim_min
)
print(f"Fine-Tuned Semantic Score (Local): {finetuned_semantic_local:.3f}")
print(
    f"This 'local' score was {finetuned_semantic_fixed - finetuned_semantic_local:.3f} lower than the fair comparison!"
)
