# 🏌️ Phân Tích Golf Swing Pro
# AI-Powered Biomechanics Analysis - Data Storm Competition 2025

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from compute_features import compute_swing_features, calculate_score
import time

mp_pose = mp.solutions.pose

# =====================================================
# CẤU HÌNH TRANG
# =====================================================
st.set_page_config(
    page_title="Phân Tích Golf Swing Pro",
    page_icon="🏌️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .score-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .score-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .tip-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        border-left: 4px solid #0ea5e9;
    }
    
    .exercise-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        border-left: 4px solid #f59e0b;
    }
    
    .badge-excellent {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-good {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-average {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-poor {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# HÀM HỖ TRỢ
# =====================================================
def extract_landmarks_from_video(video_bytes):
    """Trích xuất pose landmarks từ video"""
    tfile = "temp_video.mp4"
    with open(tfile, "wb") as f:
        f.write(video_bytes.read())
    
    cap = cv2.VideoCapture(tfile)
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5)
    
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        
        if res.pose_landmarks:
            pts = []
            for lm in res.pose_landmarks.landmark:
                pts.append([lm.x, lm.y, lm.z])
            frames.append(pts)
    
    cap.release()
    return frames

def get_score_color(score):
    if score >= 85:
        return "#10b981"
    elif score >= 70:
        return "#3b82f6"
    elif score >= 55:
        return "#f59e0b"
    else:
        return "#ef4444"

def get_score_label(score):
    if score >= 85:
        return "Xuất sắc 🏆"
    elif score >= 70:
        return "Tốt ✨"
    elif score >= 55:
        return "Trung bình 📊"
    else:
        return "Cần cải thiện 💪"

def get_badge_class(score):
    if score >= 85:
        return "badge-excellent"
    elif score >= 70:
        return "badge-good"
    elif score >= 55:
        return "badge-average"
    else:
        return "badge-poor"

def create_gauge_chart(score, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24, 'color': '#1a1a1a', 'family': 'Poppins'}},
        number={'font': {'size': 60, 'color': get_score_color(score), 'family': 'Poppins'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#667eea"},
            'bar': {'color': get_score_color(score), 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 55], 'color': 'rgba(239, 68, 68, 0.1)'},
                {'range': [55, 70], 'color': 'rgba(245, 158, 11, 0.1)'},
                {'range': [70, 85], 'color': 'rgba(59, 130, 246, 0.1)'},
                {'range': [85, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
            ],
            'threshold': {
                'line': {'color': get_score_color(score), 'width': 6},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Poppins"}
    )
    
    return fig

def create_radar_chart(detailed_scores, phase):
    metrics = []
    user_vals = []
    pro_vals = []
    
    metric_names = {
        "spine_tilt": "Nghiêng lưng", "lead_arm_angle": "Góc tay dẫn",
        "knee_flex_avg": "Gập đầu gối", "posture_stability": "Ổn định tư thế",
        "hip_rotation": "Xoay hông", "shoulder_rotation": "Xoay vai",
        "x_factor": "X-Factor", "shoulder_tilt": "Nghiêng vai",
        "hip_tilt": "Nghiêng hông", "spine_lateral_bend": "Nghiêng bên",
        "weight_shift": "Chuyển trọng tâm", "head_stability": "Ổn định đầu"
    }
    
    for metric, data in detailed_scores[phase].items():
        if metric != "phase_score" and isinstance(data, dict):
            metrics.append(metric_names.get(metric, metric.replace("_", " ").title()))
            user_vals.append(min(data["score"], 100))
            pro_vals.append(100)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=pro_vals, theta=metrics, fill='toself',
        name='Golfer Chuyên Nghiệp',
        line=dict(color='gold', width=3),
        fillcolor='rgba(255, 215, 0, 0.2)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=user_vals, theta=metrics, fill='toself',
        name='Của Bạn',
        line=dict(color='#667eea', width=3),
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    
    phase_names = {"setup": "SETUP", "top": "TOP", "impact": "IMPACT", "follow": "FOLLOW"}
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=12))),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5, font=dict(size=14, family='Poppins')),
        title=f"{phase_names.get(phase, phase.upper())} - Radar So Sánh",
        title_font=dict(size=20, family='Poppins', color='#1a1a1a'),
        height=450,
        paper_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_bar_comparison(detailed_scores, phase):
    metrics = []
    user_vals = []
    pro_vals = []
    colors = []
    
    metric_names = {
        "spine_tilt": "Nghiêng lưng", "lead_arm_angle": "Góc tay dẫn",
        "knee_flex_avg": "Gập đầu gối", "posture_stability": "Ổn định tư thế",
        "hip_rotation": "Xoay hông", "shoulder_rotation": "Xoay vai",
        "x_factor": "X-Factor", "shoulder_tilt": "Nghiêng vai",
        "hip_tilt": "Nghiêng hông", "spine_lateral_bend": "Nghiêng bên",
        "weight_shift": "Chuyển trọng tâm", "head_stability": "Ổn định đầu"
    }
    
    for metric, data in detailed_scores[phase].items():
        if metric != "phase_score" and isinstance(data, dict):
            metrics.append(metric_names.get(metric, metric.replace("_", " ").title()))
            user_vals.append(data["user"])
            pro_vals.append(data["pro"])
            colors.append(get_score_color(data["score"]))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Giá Trị Của Bạn', x=metrics, y=user_vals,
        marker=dict(color=colors, line=dict(color='white', width=2)),
        text=[f"{v:.1f}°" for v in user_vals],
        textposition='outside',
        textfont=dict(size=12, family='Poppins', color='#1a1a1a')
    ))
    
    fig.add_trace(go.Bar(
        name='Golfer Chuyên Nghiệp', x=metrics, y=pro_vals,
        marker=dict(color='gold', line=dict(color='white', width=2), pattern=dict(shape="/", solidity=0.3)),
        text=[f"{v:.1f}°" for v in pro_vals],
        textposition='outside',
        textfont=dict(size=12, family='Poppins', color='#1a1a1a')
    ))
    
    phase_names = {"setup": "SETUP", "top": "TOP", "impact": "IMPACT", "follow": "FOLLOW"}
    
    fig.update_layout(
        title=f"{phase_names.get(phase, phase.upper())} - So Sánh Chi Tiết",
        title_font=dict(size=20, family='Poppins', color='#1a1a1a'),
        xaxis_title="Chỉ Số", yaxis_title="Góc (độ)",
        barmode='group', height=450, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=14, family='Poppins')),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='Poppins'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
    )
    
    return fig

def create_phase_scores_chart(detailed_scores):
    phases = []
    scores = []
    colors = []
    
    phase_names = {"setup": "SETUP", "top": "TOP", "impact": "IMPACT", "follow": "FOLLOW"}
    
    for phase in detailed_scores:
        if "phase_score" in detailed_scores[phase]:
            phases.append(phase_names.get(phase, phase.upper()))
            score = detailed_scores[phase]["phase_score"]
            scores.append(score)
            colors.append(get_score_color(score))
    
    fig = go.Figure(go.Bar(
        x=phases, y=scores,
        marker=dict(color=colors, line=dict(color='white', width=3)),
        text=[f"{s:.1f}" for s in scores],
        textposition='outside',
        textfont=dict(size=20, family='Poppins', color='#1a1a1a', weight='bold')
    ))
    
    fig.update_layout(
        title="Điểm Số Từng Giai Đoạn",
        title_font=dict(size=24, family='Poppins', color='#1a1a1a'),
        xaxis_title="Giai Đoạn Swing", yaxis_title="Điểm",
        height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='Poppins', size=14),
        xaxis=dict(showgrid=False, tickfont=dict(size=16)),
        yaxis=dict(range=[0, 105], showgrid=True, gridcolor='rgba(0,0,0,0.05)')
    )
    
    return fig

# =====================================================
# HÀM KHUYẾN NGHỊ (giữ nguyên - đã có trong file gốc)
# =====================================================
def get_improvement_tips(metric, phase, diff):
    """Trả về khuyến nghị cụ thể cho từng metric"""
    
    tips_db = {
        "spine_tilt": {
            "title": "🔧 Cách Sửa Độ Nghiêng Lưng",
            "tips": [
                "✓ Giữ lưng thẳng từ setup đến impact, tránh cúi quá sớm",
                "✓ Tập trước gương để kiểm tra góc lưng ở mỗi phase",
                "✓ Cảm nhận sự kéo dài của cột sống, không gục người"
            ],
            "exercises": [
                "💪 **Bài tập 1:** Swing chậm với gậy trên vai, giữ lưng thẳng",
                "💪 **Bài tập 2:** Setup trước tường, lưng chạm tường nhẹ",
                "💪 **Bài tập 3:** Plank 30s x 3 lần/ngày để tăng sức lưng"
            ]
        },
        "lead_arm_angle": {
            "title": "🔧 Cách Sửa Góc Tay Dẫn",
            "tips": [
                "✓ Giữ tay trái (golfer thuận phải) thẳng trong backswing",
                "✓ Tránh gập khuỷu tay quá sớm ở top",
                "✓ Downswing: tay dẫn kéo xuống trước, tránh đẩy tay"
            ],
            "exercises": [
                "💪 **Bài tập 1:** Swing với thanh sắt dài để cảm nhận tay thẳng",
                "💪 **Bài tập 2:** Giữ gậy với 1 tay, swing chậm 20 lần",
                "💪 **Bài tập 3:** Đặt chai nước dưới nách trái, tránh rơi khi swing"
            ]
        },
        "knee_flex_avg": {
            "title": "🔧 Cách Sửa Góc Gập Đầu Gối",
            "tips": [
                "✓ Setup: Gập đầu gối nhẹ (~20-30°), không đứng thẳng cứng",
                "✓ Giữ độ gập ổn định, tránh đứng thẳng dậy ở downswing",
                "✓ Cảm nhận trọng lượng ở lòng bàn chân, không ở mũi chân"
            ],
            "exercises": [
                "💪 **Bài tập 1:** Squat nửa người 15 lần x 3 set",
                "💪 **Bài tập 2:** Wall sit 30s x 3 lần để tăng sức chân",
                "💪 **Bài tập 3:** Swing giữ 1 độ cao cố định từ đầu đến cuối"
            ]
        },
        "posture_stability": {
            "title": "🔧 Cách Tăng Độ Ổn Định Tư Thế",
            "tips": [
                "✓ Giữ chiều cao không đổi từ setup đến impact",
                "✓ Tránh nhún người lên/xuống khi swing",
                "✓ Core mạnh = tư thế ổn định"
            ],
            "exercises": [
                "💪 **Bài tập 1:** Plank 45s x 3 set",
                "💪 **Bài tập 2:** Russian twist 20 lần x 3 set",
                "💪 **Bài tập 3:** Swing trước gương, đánh dấu độ cao đầu"
            ]
        },
        "hip_rotation": {
            "title": "🔧 Cách Cải Thiện Xoay Hông",
            "tips": [
                "✓ Hông dẫn đầu trong downswing, vai theo sau",
                "✓ Backswing: Xoay hông ~45°, vai ~90°",
                "✓ Impact: Hông mở 40-45° về target"
            ],
            "exercises": [
                "💪 **Bài tập 1:** Hip rotation drill: xoay hông không xoay vai",
                "💪 **Bài tập 2:** Step drill: bước chân trái ra, xoay hông theo",
                "💪 **Bài tập 3:** Medicine ball rotation 15 lần x 3 set"
            ]
        },
        "shoulder_rotation": {
            "title": "🔧 Cách Cải Thiện Xoay Vai",
            "tips": [
                "✓ Backswing: Vai trái quay dưới cằm ~90°",
                "✓ Tránh xoay quá mức gây mất balance",
                "✓ Follow-through: Vai quay hoàn toàn về target"
            ],
            "exercises": [
                "💪 **Bài tập 1:** Cross-arm stretch 30s mỗi bên",
                "💪 **Bài tập 2:** Shoulder rotation với resistance band",
                "💪 **Bài tập 3:** Windmill exercise 10 lần mỗi bên"
            ]
        },
        "x_factor": {
            "title": "🔧 Cách Tối Ưu X-Factor",
            "tips": [
                "✓ X-Factor = hiệu số giữa xoay vai và xoay hông",
                "✓ Mục tiêu: 40-50° ở top (vai 90°, hông 45°)",
                "✓ Tạo 'dây cót' để bứt tốc độ downswing"
            ],
            "exercises": [
                "💪 **Bài tập 1:** Step-back drill: bước chân phải ra, giữ hông cố định khi xoay vai",
                "💪 **Bài tập 2:** Resistance band rotation drill",
                "💪 **Bài tập 3:** Golf-specific yoga: spinal twist"
            ]
        },
        "shoulder_tilt": {
            "title": "🔧 Cách Sửa Độ Nghiêng Vai",
            "tips": [
                "✓ Giữ 2 vai ngang nhau trong setup",
                "✓ Impact: Vai trái hơi cao hơn vai phải",
                "✓ Tránh nghiêng quá nhiều gây swing path sai"
            ],
            "exercises": [
                "💪 **Bài tập 1:** Setup với gậy ngang 2 vai, kiểm tra trước gương",
                "💪 **Bài tập 2:** One-arm plank 20s mỗi bên",
                "💪 **Bài tập 3:** Shoulder stability drill với resistance band"
            ]
        },
        "hip_tilt": {
            "title": "🔧 Cách Sửa Độ Nghiêng Hông",
            "tips": [
                "✓ Setup: 2 hông ngang nhau",
                "✓ Tránh dịch hông sang 1 bên quá sớm",
                "✓ Impact: Hông trái hơi cao hơn"
            ],
            "exercises": [
                "💪 **Bài tập 1:** Single-leg deadlift 10 lần mỗi chân",
                "💪 **Bài tập 2:** Hip hinge drill với gậy",
                "💪 **Bài tập 3:** Side plank 30s mỗi bên"
            ]
        },
        "spine_lateral_bend": {
            "title": "🔧 Cách Sửa Nghiêng Bên Lưng",
            "tips": [
                "✓ Giữ cột sống thẳng, không nghiêng sang trái/phải",
                "✓ Tránh 'reverse spine angle'",
                "✓ Impact: Lưng nghiêng nhẹ sang trái (golfer thuận phải)"
            ],
            "exercises": [
                "💪 **Bài tập 1:** Side bend stretch 15 lần mỗi bên",
                "💪 **Bài tập 2:** Bird dog exercise 10 lần mỗi bên",
                "💪 **Bài tập 3:** Swing với mirror feedback"
            ]
        },
        "weight_shift": {
            "title": "🔧 Cách Cải Thiện Chuyển Trọng Tâm",
            "tips": [
                "✓ Backswing: 60-70% trọng lượng sang chân phải",
                "✓ Downswing: Chuyển nhanh sang chân trái",
                "✓ Impact: 80-90% trọng lượng ở chân trái"
            ],
            "exercises": [
                "💪 **Bài tập 1:** Step drill: Bước chân từ phải sang trái khi swing",
                "💪 **Bài tập 2:** Swing trên 1 chân để cảm nhận balance",
                "💪 **Bài tập 3:** Pressure plate drill (nếu có thiết bị)"
            ]
        },
        "head_stability": {
            "title": "🔧 Cách Tăng Độ Ổn Định Đầu",
            "tips": [
                "✓ Giữ đầu cố định từ setup đến impact",
                "✓ Mắt nhìn bóng, tránh nhìn theo gậy quá sớm",
                "✓ Đầu chỉ quay theo sau khi bóng đã bay"
            ],
            "exercises": [
                "💪 **Bài tập 1:** Swing với bóng tennis kẹp giữa cằm và ngực",
                "💪 **Bài tập 2:** Đặt gậy trên đầu, giữ không rơi khi swing",
                "💪 **Bài tập 3:** Nhắm mắt swing để cảm nhận"
            ]
        }
    }
    
    return tips_db.get(metric, {
        "title": f"🔧 Khuyến Nghị Cho {metric.replace('_', ' ').title()}",
        "tips": [f"✓ Cần cải thiện chỉ số này. Chênh lệch: {diff:.1f}°"],
        "exercises": ["💪 Tham khảo HLV để có bài tập phù hợp"]
    })

# =====================================================
# GIAO DIỆN CHÍNH
# =====================================================

# Header
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size: 3rem;">🏌️ Phân Tích Golf Swing Chuyên Nghiệp</h1>
    <p style="margin:0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
        Phân Tích Sinh Học Chuyển Động với AI - Data Storm 2025
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar (giữ nguyên như file gốc)
with st.sidebar:
    st.markdown("## ⚙️ Cấu Hình")
    
    analysis_mode = st.radio(
        "Chọn chế độ phân tích:",
        ["📊 So sánh với Pro Baseline có sẵn", "🎯 Upload video Pro mẫu của bạn"],
        help="Chọn so sánh với baseline hoặc upload video pro riêng"
    )
    
    st.markdown("---")
    
    st.markdown("### 📹 Góc Quay")
    view_type = st.radio(
        "Chọn góc camera:",
        ["Side View (Nhìn từ bên)", "Back View (Nhìn từ phía sau)"],
        help="Chọn góc quay phù hợp với video của bạn"
    )
    
    view = "side" if "Side" in view_type else "back"
    
    st.markdown("---")
    
    st.markdown("### 📊 Thang Điểm")
    st.markdown('<div class="badge-excellent">85-100: Xuất sắc</div>', unsafe_allow_html=True)
    st.markdown('<div class="badge-good">70-84: Tốt</div>', unsafe_allow_html=True)
    st.markdown('<div class="badge-average">55-69: Trung bình</div>', unsafe_allow_html=True)
    st.markdown('<div class="badge-poor">0-54: Cần cải thiện</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📹 Lưu Ý Video")
    st.markdown("""
    - ✅ Thời lượng: 5-15 giây
    - ✅ Quay toàn thân
    - ✅ Ánh sáng tốt
    - ✅ Camera cố định (không rung)
    - ✅ Nền đơn giản
    """)

# Main Content
st.markdown("## 📤 Upload Video Golf Swing Của Bạn")
# =====================================================
# CHẾ ĐỘ 1: SO SÁNH VỚI BASELINE CÓ SẴN
# =====================================================
if analysis_mode == "📊 So sánh với Pro Baseline có sẵn":
    uploaded_file = st.file_uploader(
        "Upload video của bạn (MP4, MOV, AVI)",
        type=['mp4', 'mov', 'avi'],
        help="Upload video swing để phân tích"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.video(uploaded_file)
        
        with col2:
            st.markdown("### 🎯 Thông Tin Phân Tích")
            st.info(f"**Góc quay:** {view_type}")
            st.info(f"**Video:** {uploaded_file.name}")
            st.info(f"**Chế độ:** So sánh với Pro Baseline")
        
        if st.button("🚀 Bắt Đầu Phân Tích", type="primary", use_container_width=True):
            with st.spinner("⚙️ Đang phân tích video của bạn..."):
                progress_bar = st.progress(0)
                
                progress_bar.progress(30)
                frames = extract_landmarks_from_video(uploaded_file)
                
                if len(frames) < 10:
                    st.error("❌ Video quá ngắn hoặc không phát hiện được tư thế. Vui lòng upload video khác!")
                else:
                    progress_bar.progress(60)
                    user_features = compute_swing_features(frames, view)
                    
                    baseline_file = f"baseline_pro_{view}.json"
                    
                    try:
                        with open(baseline_file, 'r') as f:
                            baseline_features = json.load(f)
                    except:
                        st.error(f"❌ Không tìm thấy file baseline: {baseline_file}")
                        st.stop()
                    
                    progress_bar.progress(90)
                    score, detailed_scores = calculate_score(user_features, baseline_features, view)
                    progress_bar.progress(100)
                    
                    time.sleep(0.5)
                    st.success("✅ Phân tích hoàn tất! Swing của bạn đã được đánh giá chi tiết.")
                    
                    st.markdown("---")
                    st.markdown("## 🎯 KẾT QUẢ PHÂN TÍCH")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.plotly_chart(create_gauge_chart(score, "ĐIỂM TỔNG SWING"), use_container_width=True)
                    
                    with col2:
                        st.markdown("<br><br>", unsafe_allow_html=True)
                        st.markdown(f"### Đánh Giá Của Bạn")
                        st.markdown(f'<div class="{get_badge_class(score)}" style="font-size: 1.5rem; text-align: center; margin: 1rem 0;">{get_score_label(score)}</div>', unsafe_allow_html=True)
                        
                        if score >= 85:
                            st.success("Swing của bạn gần với trình độ chuyên nghiệp. Tiếp tục duy trì và luyện tập đều đặn!")
                        elif score >= 70:
                            st.info("Kỹ thuật tốt! Tập trung vào các khuyến nghị bên dưới để đạt trình độ Pro.")
                        elif score >= 55:
                            st.warning("Swing có tiềm năng. Cải thiện các điểm yếu để nâng điểm số.")
                        else:
                            st.error("Tiếp tục luyện tập! Xem phân tích chi tiết bên dưới để tập trung cải thiện.")
                    
                    st.markdown("---")
                    st.markdown("## 📈 ĐIỂM THEO GIAI ĐOẠN")
                    st.plotly_chart(create_phase_scores_chart(detailed_scores), use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown("## 🔍 CHỈ SỐ CHI TIẾT")
                    
                    for phase in detailed_scores:
                        if "phase_score" in detailed_scores[phase]:
                            phase_names = {"setup": "SETUP", "top": "TOP", "impact": "IMPACT", "follow": "FOLLOW"}
                            with st.expander(f"📊 {phase_names.get(phase, phase.upper())} - Điểm: {detailed_scores[phase]['phase_score']}/100", expanded=(phase=="impact")):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.plotly_chart(create_radar_chart(detailed_scores, phase), use_container_width=True)
                                
                                with col2:
                                    st.plotly_chart(create_bar_comparison(detailed_scores, phase), use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown("## 💡 KHUYẾN NGHỊ CẢI THIỆN (TOP 3 ƯU TIÊN)")
                    
                    priorities = []
                    metric_names = {
                        "spine_tilt": "Độ nghiêng lưng", "lead_arm_angle": "Góc tay dẫn",
                        "knee_flex_avg": "Góc gập đầu gối", "posture_stability": "Ổn định tư thế",
                        "hip_rotation": "Xoay hông", "shoulder_rotation": "Xoay vai",
                        "x_factor": "X-Factor", "shoulder_tilt": "Nghiêng vai",
                        "hip_tilt": "Nghiêng hông", "spine_lateral_bend": "Nghiêng bên lưng",
                        "weight_shift": "Chuyển trọng tâm", "head_stability": "Ổn định đầu"
                    }
                    
                    for phase in detailed_scores:
                        phase_names_full = {"setup": "SETUP", "top": "TOP", "impact": "IMPACT", "follow": "FOLLOW"}
                        for metric, data in detailed_scores[phase].items():
                            if metric != "phase_score" and isinstance(data, dict):
                                if data["score"] < 70:
                                    priorities.append({
                                        "phase": phase_names_full.get(phase, phase.upper()),
                                        "metric": metric,
                                        "metric_vn": metric_names.get(metric, metric.replace("_", " ").title()),
                                        "score": data["score"],
                                        "user": data["user"],
                                        "pro": data["pro"]
                                    })
                    
                    priorities = sorted(priorities, key=lambda x: x["score"])[:3]
                    
                    if len(priorities) == 0:
                        st.success("🎉 **Xuất sắc!** Tất cả chỉ số đều đạt mức tốt (≥70 điểm). Tiếp tục duy trì!")
                    else:
                        cols = st.columns(3)
                        for idx, item in enumerate(priorities):
                            with cols[idx]:
                                actual_diff = abs(item['user'] - item['pro'])
                                st.markdown(f"""
                                <div class="score-card" style="border-left: 4px solid {get_score_color(item['score'])};">
                                    <h4>Ưu tiên #{idx+1}</h4>
                                    <h3 style="color: {get_score_color(item['score'])};">{item['score']:.0f}/100</h3>
                                    <p><strong>{item['metric_vn']}</strong></p>
                                    <p style="font-size: 0.9rem; color: #666;">
                                        Giai đoạn: {item['phase']}<br>
                                        Chênh lệch: {actual_diff:.1f}°
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        st.markdown("## 📋 HƯỚNG DẪN CẢI THIỆN CHI TIẾT")
                        
                        for idx, item in enumerate(priorities):
                            actual_diff = abs(item['user'] - item['pro'])
                            tips = get_improvement_tips(item['metric'], item['phase'], actual_diff)
                            
                            with st.expander(f"🎯 Ưu tiên #{idx+1}: {item['metric_vn']} ({item['phase']}) - {item['score']:.0f}/100", expanded=(idx==0)):
                                st.markdown(f"### {tips['title']}")
                                
                                st.markdown("#### 📌 Các Điểm Cần Lưu Ý:")
                                for tip in tips['tips']:
                                    st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)
                                
                                st.markdown("#### 💪 Bài Tập Cải Thiện:")
                                for exercise in tips['exercises']:
                                    st.markdown(f'<div class="exercise-box">{exercise}</div>', unsafe_allow_html=True)
                    
                    # EXPORT BÁO CÁO - ĐÃ FIX
                    st.markdown("---")
                    st.markdown("## 📥 TẢI BÁO CÁO")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        report = {
                            "diem_tong": score,
                            "goc_quay": view,
                            "chi_tiet": detailed_scores
                        }
                        st.download_button(
                            "📄 Tải Báo Cáo JSON",
                            data=json.dumps(report, indent=2, ensure_ascii=False),
                            file_name=f"phan_tich_golf_{view}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    with col2:
                        phase_names = {"setup": "SETUP", "top": "TOP", "impact": "IMPACT", "follow": "FOLLOW"}
                        summary = f"""
=== BÁO CÁO PHÂN TÍCH GOLF SWING ===
Góc quay: {view_type}
Điểm tổng: {score}/100
Đánh giá: {get_score_label(score)}

=== ĐIỂM THEO GIAI ĐOẠN ===
"""
                        for phase in detailed_scores:
                            if "phase_score" in detailed_scores[phase]:
                                summary += f"{phase_names.get(phase, phase.upper())}: {detailed_scores[phase]['phase_score']}/100\n"
                        
                        st.download_button(
                            "📝 Tải Tóm Tắt Text",
                            data=summary,
                            file_name=f"tom_tat_golf_{view}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )

# =====================================================
# CHẾ ĐỘ 2: UPLOAD 2 VIDEO
# =====================================================
else:
    st.info("🎯 **Chế độ Tùy Chỉnh:** Upload cả video của bạn và video Pro mẫu để so sánh trực tiếp!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Video Của Bạn")
        user_video = st.file_uploader(
            "Upload video swing của bạn",
            type=['mp4', 'mov', 'avi'],
            key="user_video"
        )
        if user_video:
            st.video(user_video)
    
    with col2:
        st.markdown("### 🏆 Video Pro Mẫu")
        pro_video = st.file_uploader(
            "Upload video Pro mẫu để so sánh",
            type=['mp4', 'mov', 'avi'],
            key="pro_video"
        )
        if pro_video:
            st.video(pro_video)
    
    if user_video and pro_video:
        if st.button("🚀 Phân Tích & So Sánh", type="primary", use_container_width=True):
            with st.spinner("⚙️ Đang phân tích cả 2 video..."):
                progress_bar = st.progress(0)
                
                progress_bar.progress(20)
                st.info("📊 Đang xử lý video của bạn...")
                user_frames = extract_landmarks_from_video(user_video)
                
                progress_bar.progress(50)
                st.info("🏆 Đang xử lý video Pro mẫu...")
                pro_frames = extract_landmarks_from_video(pro_video)
                
                if len(user_frames) < 10 or len(pro_frames) < 10:
                    st.error("❌ Một trong 2 video quá ngắn hoặc không phát hiện được tư thế!")
                else:
                    progress_bar.progress(70)
                    user_features = compute_swing_features(user_frames, view)
                    pro_features = compute_swing_features(pro_frames, view)
                    
                    progress_bar.progress(90)
                    score, detailed_scores = calculate_score(user_features, pro_features, view)
                    progress_bar.progress(100)
                    
                    time.sleep(0.5)
                    st.success("✅ Phân tích hoàn tất! Đã so sánh 2 video thành công!")
                    
                    st.markdown("---")
                    st.markdown("## 🎯 KẾT QUẢ PHÂN TÍCH")
                    st.info("📌 **Lưu ý:** Bạn đang so sánh với video Pro mẫu đã upload, không phải baseline có sẵn!")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.plotly_chart(create_gauge_chart(score, "ĐIỂM TỔNG SWING"), use_container_width=True)
                    
                    with col2:
                        st.markdown("<br><br>", unsafe_allow_html=True)
                        st.markdown(f"### Đánh Giá Của Bạn")
                        st.markdown(f'<div class="{get_badge_class(score)}" style="font-size: 1.5rem; text-align: center; margin: 1rem 0;">{get_score_label(score)}</div>', unsafe_allow_html=True)
                        
                        if score >= 85:
                            st.success("Swing của bạn rất gần với mẫu Pro. Xuất sắc!")
                        elif score >= 70:
                            st.info("Kỹ thuật tốt! Tập trung vào các khuyến nghị bên dưới.")
                        elif score >= 55:
                            st.warning("Swing có tiềm năng. Cải thiện các điểm được gợi ý.")
                        else:
                            st.error("Tiếp tục luyện tập! Xem phân tích chi tiết bên dưới.")
                    
                    st.markdown("---")
                    st.markdown("## 📈 ĐIỂM THEO GIAI ĐOẠN")
                    st.plotly_chart(create_phase_scores_chart(detailed_scores), use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown("## 🔍 CHỈ SỐ CHI TIẾT")
                    
                    for phase in detailed_scores:
                        if "phase_score" in detailed_scores[phase]:
                            phase_names = {"setup": "SETUP", "top": "TOP", "impact": "IMPACT", "follow": "FOLLOW"}
                            with st.expander(f"📊 {phase_names.get(phase, phase.upper())} - Điểm: {detailed_scores[phase]['phase_score']}/100", expanded=(phase=="impact")):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.plotly_chart(create_radar_chart(detailed_scores, phase), use_container_width=True)
                                
                                with col2:
                                    st.plotly_chart(create_bar_comparison(detailed_scores, phase), use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown("## 💡 KHUYẾN NGHỊ CẢI THIỆN (TOP 3 ƯU TIÊN)")
                    
                    priorities = []
                    metric_names = {
                        "spine_tilt": "Độ nghiêng lưng", "lead_arm_angle": "Góc tay dẫn",
                        "knee_flex_avg": "Góc gập đầu gối", "posture_stability": "Ổn định tư thế",
                        "hip_rotation": "Xoay hông", "shoulder_rotation": "Xoay vai",
                        "x_factor": "X-Factor", "shoulder_tilt": "Nghiêng vai",
                        "hip_tilt": "Nghiêng hông", "spine_lateral_bend": "Nghiêng bên lưng",
                        "weight_shift": "Chuyển trọng tâm", "head_stability": "Ổn định đầu"
                    }
                    
                    for phase in detailed_scores:
                        phase_names_full = {"setup": "SETUP", "top": "TOP", "impact": "IMPACT", "follow": "FOLLOW"}
                        for metric, data in detailed_scores[phase].items():
                            if metric != "phase_score" and isinstance(data, dict):
                                if data["score"] < 70:
                                    priorities.append({
                                        "phase": phase_names_full.get(phase, phase.upper()),
                                        "metric": metric,
                                        "metric_vn": metric_names.get(metric, metric.replace("_", " ").title()),
                                        "score": data["score"],
                                        "user": data["user"],
                                        "pro": data["pro"]
                                    })
                    
                    priorities = sorted(priorities, key=lambda x: x["score"])[:3]
                    
                    if len(priorities) == 0:
                        st.success("🎉 **Xuất sắc!** Tất cả chỉ số đều đạt mức tốt!")
                    else:
                        cols = st.columns(3)
                        for idx, item in enumerate(priorities):
                            with cols[idx]:
                                actual_diff = abs(item['user'] - item['pro'])
                                st.markdown(f"""
                                <div class="score-card" style="border-left: 4px solid {get_score_color(item['score'])};">
                                    <h4>Ưu tiên #{idx+1}</h4>
                                    <h3 style="color: {get_score_color(item['score'])};">{item['score']:.0f}/100</h3>
                                    <p><strong>{item['metric_vn']}</strong></p>
                                    <p style="font-size: 0.9rem; color: #666;">
                                        Giai đoạn: {item['phase']}<br>
                                        Chênh lệch: {actual_diff:.1f}°
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        st.markdown("## 📋 HƯỚNG DẪN CẢI THIỆN CHI TIẾT")
                        
                        for idx, item in enumerate(priorities):
                            actual_diff = abs(item['user'] - item['pro'])
                            tips = get_improvement_tips(item['metric'], item['phase'], actual_diff)
                            
                            with st.expander(f"🎯 Ưu tiên #{idx+1}: {item['metric_vn']} ({item['phase']}) - {item['score']:.0f}/100", expanded=(idx==0)):
                                st.markdown(f"### {tips['title']}")
                                
                                st.markdown("#### 📌 Các Điểm Cần Lưu Ý:")
                                for tip in tips['tips']:
                                    st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)
                                
                                st.markdown("#### 💪 Bài Tập Cải Thiện:")
                                for exercise in tips['exercises']:
                                    st.markdown(f'<div class="exercise-box">{exercise}</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p style="margin: 0; font-size: 0.9rem;">⛳ Phân Tích Golf Swing Pro v2.0</p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem;">
        Phát triển bởi <strong>Lâm Tuấn Vũ • Nguyễn Vũ Thắng • Đỗ Gia Khiêm</strong> (VTK Team)
    </p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem;">
        Data Storm Competition 2025 | Phân Tích Sinh Học Chuyển Động với AI
    </p>
</div>
""", unsafe_allow_html=True)
