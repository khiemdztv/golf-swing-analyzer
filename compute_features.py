import math
import numpy as np

def angle_3d(a, b, c):
    """Tính góc 3D chính xác hơn"""
    ba = np.array([a[0]-b[0], a[1]-b[1], a[2]-b[2]])
    bc = np.array([c[0]-b[0], c[1]-b[1], c[2]-b[2]])
    dot = np.dot(ba, bc)
    mag1 = np.linalg.norm(ba)
    mag2 = np.linalg.norm(bc)
    if mag1 * mag2 == 0:
        return 0
    cos_angle = dot / (mag1 * mag2)
    cos_angle = np.clip(cos_angle, -1, 1)
    return math.degrees(math.acos(cos_angle))

def angle_2d(a, b, c):
    """Tính góc 2D cho shoulder/hip tilt"""
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag1 = math.sqrt(ba[0]**2 + ba[1]**2)
    mag2 = math.sqrt(bc[0]**2 + bc[1]**2)
    if mag1*mag2 == 0:
        return 0
    cos_angle = dot/(mag1*mag2)
    cos_angle = max(-1, min(1, cos_angle))
    return math.degrees(math.acos(cos_angle))

def compute_features_frame(points, view_type="side"):
    """Tính các chỉ số biomechanics chi tiết"""
    # Landmarks cơ bản
    nose = points[0]
    l_shoulder = points[11]
    r_shoulder = points[12]
    l_elbow = points[13]
    r_elbow = points[14]
    l_wrist = points[15]
    r_wrist = points[16]
    l_hip = points[23]
    r_hip = points[24]
    l_knee = points[25]
    r_knee = points[26]
    l_ankle = points[27]
    r_ankle = points[28]
    
    features = {}
    
    if view_type == "side":
        # 1. Spine angle (góc nghiêng lưng)
        mid_shoulder = [(l_shoulder[i] + r_shoulder[i])/2 for i in range(3)]
        mid_hip = [(l_hip[i] + r_hip[i])/2 for i in range(3)]
        spine_angle = abs(90 - angle_2d(nose, mid_shoulder, mid_hip))
        features["spine_tilt"] = spine_angle
        
        # 2. Hip rotation (xoay hông)
        hip_rotation = angle_2d(l_hip, mid_hip, r_hip)
        features["hip_rotation"] = hip_rotation
        
        # 3. Shoulder rotation (xoay vai)
        shoulder_rotation = angle_2d(l_shoulder, mid_shoulder, r_shoulder)
        features["shoulder_rotation"] = shoulder_rotation
        
        # 4. Lead arm angle (góc tay dẫn - quan trọng!)
        lead_arm = angle_3d(l_shoulder, l_elbow, l_wrist)
        features["lead_arm_angle"] = lead_arm
        
        # 5. Trail arm angle
        trail_arm = angle_3d(r_shoulder, r_elbow, r_wrist)
        features["trail_arm_angle"] = trail_arm
        
        # 6. Hip-shoulder separation (X-factor)
        x_factor = abs(shoulder_rotation - hip_rotation)
        features["x_factor"] = x_factor
        
        # 7. Knee flex (góc gập đầu gối)
        l_knee_flex = angle_3d(l_hip, l_knee, l_ankle)
        r_knee_flex = angle_3d(r_hip, r_knee, r_ankle)
        features["knee_flex_avg"] = (l_knee_flex + r_knee_flex) / 2
        
        # 8. Posture Stability - GIỮ NGUYÊN CÔNG THỨC CŨ (nhân 100)
        posture_height = abs(mid_hip[1] - (l_ankle[1] + r_ankle[1])/2)
        features["posture_stability"] = posture_height * 100  # ← NHÂN 100 (giống baseline cũ)
        
    else:  # back view
        # 1. Shoulder tilt (GÓC - degrees)
        shoulder_tilt = abs(90 - angle_2d(l_hip, l_shoulder, r_shoulder))
        features["shoulder_tilt"] = shoulder_tilt
        
        # 2. Hip tilt (GÓC - degrees)
        hip_tilt = abs(90 - angle_2d(l_shoulder, l_hip, r_hip))
        features["hip_tilt"] = hip_tilt
        
        # 3. Spine lateral bend (KHOẢNG CÁCH tương đối)
        mid_shoulder = [(l_shoulder[i] + r_shoulder[i])/2 for i in range(3)]
        mid_hip = [(l_hip[i] + r_hip[i])/2 for i in range(3)]
        spine_lateral = abs(mid_shoulder[0] - mid_hip[0])
        features["spine_lateral_bend"] = spine_lateral
        
        # 4. Weight shift (KHOẢNG CÁCH tương đối)
        weight_dist = abs(l_hip[0] - r_hip[0])
        features["weight_shift"] = weight_dist
        
        # 5. Head stability (KHOẢNG CÁCH tương đối)
        head_center = abs(nose[0] - (l_shoulder[0] + r_shoulder[0])/2)
        features["head_stability"] = head_center
    
    return features

def detect_swing_phases(frames):
    """Tự động phát hiện các phase của swing"""
    if len(frames) < 20:
        return None
    
    # Tính hip rotation cho mỗi frame để tìm top và impact
    hip_rotations = []
    for pts in frames:
        l_hip = pts[23]
        r_hip = pts[24]
        mid_hip = [(l_hip[i] + r_hip[i])/2 for i in range(3)]
        rotation = angle_2d(l_hip, mid_hip, r_hip)
        hip_rotations.append(rotation)
    
    # Smooth signal
    hip_rotations = np.convolve(hip_rotations, np.ones(5)/5, mode='same')
    
    # Setup: frame đầu tiên stable (10% đầu)
    setup_idx = int(len(frames) * 0.1)
    
    # Top: max hip rotation (backswing peak)
    top_idx = np.argmax(hip_rotations[:int(len(frames)*0.6)])
    
    # Impact: điểm hip rotation về gần ban đầu (sau top)
    target_rotation = hip_rotations[setup_idx]
    impact_candidates = hip_rotations[top_idx:]
    impact_idx = top_idx + np.argmin(np.abs(impact_candidates - target_rotation))
    
    # Follow-through: 80% của video
    follow_idx = int(len(frames) * 0.8)
    
    return {
        "setup": setup_idx,
        "top": top_idx,
        "impact": impact_idx,
        "follow": follow_idx
    }

def compute_swing_features(frames, view_type="side"):
    """Tính features cho toàn bộ swing với phase detection"""
    phases_idx = detect_swing_phases(frames)
    if phases_idx is None:
        return None
    
    features = {}
    for phase_name, idx in phases_idx.items():
        if idx < len(frames):
            features[phase_name] = compute_features_frame(frames[idx], view_type)
    
    return features

def calculate_score(user_features, baseline_features, view_type="side"):
    """Tính điểm tổng 100 với tolerance đã tối ưu"""
    
    if view_type == "side":
        weights = {
            "setup": {
                "spine_tilt": (0.30, 5),
                "lead_arm_angle": (0.25, 10),
                "knee_flex_avg": (0.20, 12),
                "posture_stability": (0.15, 15),       # ← TĂNG tolerance lên 15
                "hip_rotation": (0.10, 8),
            },
            "top": {
                "x_factor": (0.30, 10),
                "shoulder_rotation": (0.25, 12),
                "spine_tilt": (0.20, 8),
                "lead_arm_angle": (0.15, 12),
                "knee_flex_avg": (0.10, 15),
            },
            "impact": {
                "hip_rotation": (0.30, 10),
                "spine_tilt": (0.25, 6),
                "lead_arm_angle": (0.20, 10),
                "posture_stability": (0.15, 15),       # ← TĂNG tolerance lên 15
                "x_factor": (0.10, 12),
            },
            "follow": {
                "shoulder_rotation": (0.30, 15),
                "spine_tilt": (0.25, 10),
                "lead_arm_angle": (0.20, 15),
                "posture_stability": (0.15, 20),       # ← TĂNG tolerance lên 20
                "knee_flex_avg": (0.10, 18),
            }
        }
    else:  # back view
        weights = {
            "setup": {
                "shoulder_tilt": (0.30, 6),
                "hip_tilt": (0.25, 6),
                "spine_lateral_bend": (0.20, 0.08),
                "head_stability": (0.15, 0.08),
                "weight_shift": (0.10, 0.15),
            },
            "top": {
                "shoulder_tilt": (0.35, 10),
                "weight_shift": (0.30, 0.15),
                "spine_lateral_bend": (0.20, 0.10),
                "hip_tilt": (0.15, 10),
            },
            "impact": {
                "shoulder_tilt": (0.30, 8),
                "hip_tilt": (0.25, 8),
                "weight_shift": (0.25, 0.15),
                "head_stability": (0.20, 0.10),
            },
            "follow": {
                "shoulder_tilt": (0.35, 12),
                "spine_lateral_bend": (0.25, 0.12),
                "weight_shift": (0.20, 0.20),
                "head_stability": (0.20, 0.12),
            }
        }
    
    total_score = 0
    detailed_scores = {}
    valid_phases = 0
    
    for phase in user_features:
        if phase not in baseline_features or phase not in weights:
            continue
        
        phase_score = 0
        detailed_scores[phase] = {}
        
        for metric, (weight, tolerance) in weights[phase].items():
            if metric not in user_features[phase] or metric not in baseline_features[phase]:
                continue
            
            user_val = user_features[phase][metric]
            base_val = baseline_features[phase][metric]
            diff = abs(user_val - base_val)
            
            # CÔNG THỨC CHẤM ĐIỂM GRADIENT
            if diff <= tolerance:
                metric_score = 100
            elif diff <= tolerance * 2:
                metric_score = 100 - (diff - tolerance) / tolerance * 30
            elif diff <= tolerance * 3:
                metric_score = 70 - (diff - tolerance * 2) / tolerance * 30
            else:
                metric_score = max(0, 40 - (diff - tolerance * 3) / tolerance * 10)
            
            metric_score = max(0, min(100, metric_score))
            phase_score += metric_score * weight
            
            detailed_scores[phase][metric] = {
                "user": round(user_val, 2),
                "pro": round(base_val, 2),
                "diff": round(diff, 2),
                "score": round(metric_score, 1)
            }
        
        detailed_scores[phase]["phase_score"] = round(phase_score, 1)
        total_score += phase_score
        valid_phases += 1
    
    final_score = total_score / valid_phases if valid_phases > 0 else 0
    return round(final_score, 1), detailed_scores
