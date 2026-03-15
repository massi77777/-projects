import streamlit as st
import numpy as np
import pickle

st.set_page_config(
    page_title="نظام التنبؤ بمرض السكري",
    page_icon="🩺",
    layout="centered"
)

# ── تحميل النموذج ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, scaler, features

model, scaler, FEATURE_NAMES = load_model()

# ── واجهة المستخدم ─────────────────────────────────────────────────────────
st.title("🩺 نظام التنبؤ بمرض السكري")
st.markdown(
    "أداة للتنبؤ المبكر بمرض السكري باستخدام الذكاء الاصطناعي. "
    "اضبط القيم ثم اضغط **تحليل**."
)
st.markdown("---")

FEATURE_INFO = {
    "Pregnancies":              ("عدد مرات الحمل",               0,   20,  3,    1),
    "Glucose":                  ("مستوى السكر في الدم (mg/dL)",  0,  300, 117,  1),
    "BloodPressure":            ("ضغط الدم الانبساطي (mmHg)",    0,  150,  72,  1),
    "SkinThickness":            ("سماكة طية الجلد (mm)",         0,  100,  23,  1),
    "Insulin":                  ("مستوى الأنسولين (mu U/ml)",    0,  900,  30,  1),
    "BMI":                      ("مؤشر كتلة الجسم (BMI)",        0.0, 70.0, 32.0, 0.1),
    "DiabetesPedigreeFunction": ("معامل الوراثة السكرية",        0.0,  3.0,  0.3, 0.01),
    "Age":                      ("العمر (سنة)",                   1,  120,  33,  1),
}

# ── الـ Sliders في عمودين ──────────────────────────────────────────────────
col1, col2 = st.columns(2)
values = {}

for i, feat in enumerate(FEATURE_NAMES):
    if feat in FEATURE_INFO:
        label, mn, mx, default, step = FEATURE_INFO[feat]
        col = col1 if i % 2 == 0 else col2
        if isinstance(step, float):
            values[feat] = col.slider(label, float(mn), float(mx), float(default), step)
        else:
            values[feat] = col.slider(label, int(mn), int(mx), int(default), step)
    else:
        values[feat] = st.number_input(feat, value=0.0)

st.markdown("---")

# ── زر التحليل ────────────────────────────────────────────────────────────
if st.button("🔍 تحليل", use_container_width=True, type="primary"):
    input_data   = np.array([values[f] for f in FEATURE_NAMES]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = (
        model.predict_proba(input_scaled)[0]
        if hasattr(model, "predict_proba") else None
    )

    st.markdown("### 📋 النتيجة")

    if prediction == 1:
        st.error("🔴 **النتيجة: مريض بالسكري** — مستوى الخطر: مرتفع")
        st.warning("⚠️ يُنصح بزيارة الطبيب فوراً وإجراء الفحوصات اللازمة.")
    else:
        st.success("🟢 **النتيجة: لا يعاني من السكري** — مستوى الخطر: منخفض")
        st.info("✅ استمر في نمط حياتك الصحي والحفاظ على وزن مثالي.")

    if probability is not None:
        pct = float(probability[1])
        st.markdown(f"**📊 احتمالية الإصابة: `{pct:.1%}`**")
        st.progress(pct)

    st.caption("⚕️ هذه الأداة للأغراض التعليمية فقط ولا تُغني عن استشارة طبيب متخصص.")
