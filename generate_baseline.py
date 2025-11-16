import json
import os
import numpy as np
from compute_features import compute_swing_features

def generate_baseline(folder, output_file, view_type="side"):
    """Generate baseline với outlier removal"""
    data_list = []
    
    print(f"\n{'='*50}")
    print(f"Generating baseline for {view_type.upper()} view")
    print(f"{'='*50}\n")

    # Load tất cả json pose files
    for f in os.listdir(folder):
        if f.endswith(".json"):
            filepath = os.path.join(folder, f)
            print(f"📂 Reading: {f}")
            
            try:
                with open(filepath, 'r') as file:
                    frames = json.load(file)
                
                # Compute features với view type
                feat = compute_swing_features(frames, view_type)
                
                if feat is not None:
                    data_list.append(feat)
                    print(f"   ✅ Extracted {len(frames)} frames")
                else:
                    print(f"   ⚠️  Could not extract features (too short)")
                    
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")

    if not data_list:
        print("\n❌ No valid data found!")
        return

    print(f"\n✅ Successfully processed {len(data_list)} videos")
    print(f"\nCalculating baseline with outlier removal...\n")

    # Tính baseline với median (robust hơn mean)
    baseline = {}
    
    # Lấy tất cả phases
    all_phases = set()
    for data in data_list:
        all_phases.update(data.keys())
    
    for phase in all_phases:
        baseline[phase] = {}
        
        # Lấy tất cả features trong phase này
        phase_data = [d[phase] for d in data_list if phase in d]
        
        if not phase_data:
            continue
        
        # Lấy tất cả metrics
        all_metrics = set()
        for pd in phase_data:
            all_metrics.update(pd.keys())
        
        for metric in all_metrics:
            # Lấy tất cả giá trị của metric này
            values = [pd[metric] for pd in phase_data if metric in pd]
            
            if not values:
                continue
            
            values = np.array(values)
            
            # Loại bỏ outliers bằng IQR method
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Lọc values
            filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]
            
            if len(filtered_values) == 0:
                filtered_values = values  # Nếu lọc hết thì giữ nguyên
            
            # Dùng median thay vì mean (robust hơn)
            baseline[phase][metric] = float(np.median(filtered_values))
            
            print(f"  {phase}.{metric}:")
            print(f"    Mean: {np.mean(values):.2f}°")
            print(f"    Median: {np.median(values):.2f}°")
            print(f"    Baseline (filtered): {baseline[phase][metric]:.2f}°")
            print(f"    Removed {len(values) - len(filtered_values)} outliers")

    # Save baseline
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"✅ Saved baseline: {output_file}")
    print(f"{'='*50}\n")
    
    # Print summary
    print("📊 BASELINE SUMMARY:")
    for phase in baseline:
        print(f"\n{phase.upper()}:")
        for metric, value in baseline[phase].items():
            print(f"  {metric}: {value:.2f}")


def validate_baseline(baseline_file):
    """Kiểm tra baseline có hợp lý không"""
    print(f"\n🔍 Validating {baseline_file}...")
    
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    
    issues = []
    
    for phase, metrics in baseline.items():
        for metric, value in metrics.items():
            # Check for unrealistic values
            if value < 0 or value > 180:
                issues.append(f"⚠️  {phase}.{metric} = {value:.2f}° (out of range 0-180)")
            
            # Check for suspiciously low values
            if value < 5 and 'angle' in metric:
                issues.append(f"⚠️  {phase}.{metric} = {value:.2f}° (suspiciously low)")
    
    if issues:
        print("\n❌ Found issues:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ Baseline looks good!")
    
    return len(issues) == 0


if __name__ == "__main__":
    # Đường dẫn folder chứa video pro đã extract
    # Thay đổi đường dẫn này theo máy của bạn
    side_folder = r"D:\Documents\Data Storm\video vdv pro\sideview"
    back_folder = r"D:\Documents\Data Storm\video vdv pro\backview"
    
    # Generate baselines
    print("🏌️ GENERATING PRO BASELINES")
    
    # Side view
    if os.path.exists(side_folder):
        generate_baseline(side_folder, "baseline_pro_side.json", view_type="side")
        validate_baseline("baseline_pro_side.json")
    else:
        print(f"⚠️  Folder not found: {side_folder}")
    
    # Back view
    if os.path.exists(back_folder):
        generate_baseline(back_folder, "baseline_pro_back.json", view_type="back")
        validate_baseline("baseline_pro_back.json")
    else:
        print(f"⚠️  Folder not found: {back_folder}")
    
    print("\n🎉 DONE!")