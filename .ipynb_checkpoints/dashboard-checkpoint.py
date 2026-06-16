import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from fpdf import FPDF

# ============================================
# تحميل النموذج
# ============================================
model = joblib.load("xgboost_hotel_model.pkl")

# ============================================
# تحميل البيانات
# ============================================
df = pd.read_csv("data/Hotels_cleaned.csv")

# ============================================
# Streamlit Theme
# ============================================
st.markdown("""
    <style>
        .stApp { background-color: #F5F7FA; }
        h1, h2, h3 { color: #1A5276; }
        .stButton>button {
            background-color: #1A5276; color: white;
            border-radius: 8px; padding: 0.6rem 1.2rem; font-size: 16px;
        }
        .stButton>button:hover { background-color: #154360; }
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            border-radius: 6px; border: 1px solid #AAB7B8;
        }
        section[data-testid="stSidebar"] { background-color: #EBF5FB; }
    </style>
""", unsafe_allow_html=True)

# ============================================
# Sidebar - اختيار الصفحة
# ============================================
page = st.sidebar.selectbox(
    "📌 اختر الصفحة",
    [
        "📊 تحليل البيانات",
        "🔮 تنبؤ فردي",
        "📂 تنبؤ جماعي (CSV)",
        "📘 Insights التقرير",
        "📊 Executive Dashboard"
    ]
)

# ============================================
# عنوان التطبيق
# ============================================
st.title("Hotel Booking Cancellation Prediction Dashboard")
st.write("تحليل وتوقع إلغاء الحجوزات باستخدام XGBoost")

# ============================================
# تجهيز البيانات للنموذج
# ============================================
df_model = df.copy()
df_model = df_model.drop(columns=['reservation_status', 'reservation_status_date'], errors='ignore')

for col in df_model.select_dtypes(include='object').columns:
    df_model[col] = LabelEncoder().fit_transform(df_model[col].astype(str))

X = df_model.drop('is_canceled', axis=1)
y = df_model['is_canceled']

# ============================================
# صفحة تحليل البيانات
# ============================================
if page == "📊 تحليل البيانات":

    st.markdown("""
    <div style="background-color: white; padding: 20px; border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px;">
    <h3 style="color:#1A5276;">📈 نظرة عامة على أداء النموذج</h3>
    <p>في هذا القسم نعرض أداء نموذج XGBoost على كامل البيانات.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.checkbox("عرض أول 5 صفوف"):
        st.dataframe(df.head())

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    st.subheader("📌 Model Accuracy")
    st.write(f"Accuracy: **{acc:.4f}**")

    st.subheader("📌 Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.subheader("📌 Classification Report")
    st.text(classification_report(y, y_pred))

    st.subheader("📌 Feature Importance")
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
    st.bar_chart(feat_imp.set_index('Feature'))

    st.subheader("📥 Download Predictions")
    results_df = df.copy()
    results_df["prediction"] = y_pred
    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button("تنزيل النتائج بصيغة CSV", csv_data, "hotel_predictions.csv", "text/csv")

# ============================================
# صفحة التنبؤ الفردي
# ============================================
if page == "🔮 تنبؤ فردي":

    st.markdown("""
    <div style="background-color: white; padding: 25px; border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px;">
    <h2 style="color:#1A5276;">🔮 التنبؤ بإلغاء حجز واحد</h2>
    <p>أدخل بيانات الحجز وسيقوم النموذج بإعطاء التوقع مباشرة.</p>
    </div>
    """, unsafe_allow_html=True)

    lead_time = st.number_input("Lead Time", min_value=0, max_value=500)
    adr = st.number_input("ADR", min_value=0.0, max_value=1000.0)
    adults = st.number_input("عدد البالغين", min_value=1, max_value=10)
    children = st.number_input("عدد الأطفال", min_value=0, max_value=10)
    babies = st.number_input("عدد الرضع", min_value=0, max_value=10)
    previous_cancellations = st.number_input("عدد الإلغاءات السابقة", min_value=0, max_value=20)
    previous_bookings_not_canceled = st.number_input("عدد الحجوزات السابقة غير الملغاة", min_value=0, max_value=50)
    booking_changes = st.number_input("عدد التغييرات في الحجز", min_value=0, max_value=20)
    total_of_special_requests = st.number_input("عدد الطلبات الخاصة", min_value=0, max_value=10)

    if st.button("تنفيذ التنبؤ"):

        input_data = pd.DataFrame([{
            "lead_time": lead_time,
            "adr": adr,
            "adults": adults,
            "children": children,
            "babies": babies,
            "previous_cancellations": previous_cancellations,
            "previous_bookings_not_canceled": previous_bookings_not_canceled,
            "booking_changes": booking_changes,
            "total_of_special_requests": total_of_special_requests
        }])

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("❌ التوقع: سيتم **إلغاء الحجز**")
        else:
            st.success("✅ التوقع: **لن يتم إلغاء الحجز**")

# ============================================
# صفحة التنبؤ الجماعي
# ============================================
if page == "📂 تنبؤ جماعي (CSV)":

    st.header("📂 التنبؤ الجماعي من خلال رفع ملف CSV")
    st.write("قم برفع ملف يحتوي على بيانات حجوزات متعددة وسيقوم النموذج بإعطاء التوقع لكل صف.")

    uploaded_file = st.file_uploader("ارفع ملف CSV", type=["csv"])

    if uploaded_file is not None:

        input_df = pd.read_csv(uploaded_file)
        st.subheader("📌 البيانات المرفوعة")
        st.dataframe(input_df.head())

        required_cols = [
            "lead_time", "adr", "adults", "children", "babies",
            "previous_cancellations", "previous_bookings_not_canceled",
            "booking_changes", "total_of_special_requests"
        ]

        missing_cols = [col for col in required_cols if col not in input_df.columns]

        if missing_cols:
            st.error(f"❌ الأعمدة التالية ناقصة: {missing_cols}")
        else:
            predictions = model.predict(input_df)
            input_df["prediction"] = predictions

            st.subheader("📌 نتائج التنبؤ")
            st.dataframe(input_df)

            csv_output = input_df.to_csv(index=False).encode("utf-8")
            st.download_button("تنزيل النتائج", csv_output, "batch_predictions.csv", "text/csv")
# ============================================
# صفحة Insights الجاهزة للتقرير
# ============================================
if page == "📘 Insights التقرير":

    st.markdown("""
    <div style="
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;">
    <h2 style="color:#1A5276;">📘 تقرير التحليلات الشامل</h2>
    <p>هذا التقرير يقدّم ملخصًا تنفيذيًا لأهم النتائج المستخلصة من بيانات الحجوزات.</p>
    </div>
    """, unsafe_allow_html=True)

    # ============================
    # Executive Summary
    # ============================
    st.subheader("📌 الملخص التنفيذي (Executive Summary)")
    st.write("""
    - نسبة الإلغاء العامة مرتفعة مقارنة بالحجوزات المكتملة.
    - العملاء ذوو **Lead Time طويل** لديهم احتمال أعلى للإلغاء.
    - الحجوزات ذات **ADR منخفض** ترتبط بإلغاء أعلى.
    - وجود **طلبات خاصة قليلة** يشير إلى احتمال أعلى للإلغاء.
    """)

    st.markdown("---")

    # ============================
    # KPIs
    # ============================
    st.subheader("📊 المؤشرات الرئيسية (KPIs)")

    total_bookings = len(df)
    cancel_rate = df['is_canceled'].mean() * 100
    avg_adr = df['adr'].mean()
    avg_lead = df['lead_time'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("إجمالي الحجوزات", total_bookings)
    col2.metric("نسبة الإلغاء", f"{cancel_rate:.2f}%")
    col3.metric("متوسط ADR", f"{avg_adr:.2f}")
    col4.metric("متوسط Lead Time", f"{avg_lead:.1f} يوم")

    st.markdown("---")

    # ============================
    # Top Insights
    # ============================
    st.subheader("💡 أهم الاستنتاجات (Top Insights)")

    st.write("""
    - **Lead Time** هو أقوى مؤشر على الإلغاء.
    - العملاء الذين لديهم **إلغاءات سابقة** يميلون لإلغاء الحجز مرة أخرى.
    - الحجوزات ذات **عدد بالغين أكبر** غالبًا لا تُلغى.
    - **الطلبات الخاصة** تعكس جدية العميل، وكلما زادت قلّ احتمال الإلغاء.
    """)

    st.markdown("---")

    # ============================
    # Reasons Behind Cancellations
    # ============================
    st.subheader("❗ أسباب الإلغاء الأكثر شيوعًا")

    st.write("""
    - تغيّر خطط السفر بسبب طول المدة بين الحجز والوصول.
    - الأسعار المنخفضة غالبًا ترتبط بحجوزات غير جدية.
    - العملاء الذين لديهم تاريخ سابق من الإلغاء يستمرون في نفس السلوك.
    """)

    st.markdown("---")

    # ============================
    # Recommendations
    # ============================
    st.subheader("🛠️ التوصيات (Recommendations)")

    st.write("""
    - تقديم **عروض خاصة** للعملاء ذوي Lead Time طويل لتقليل الإلغاء.
    - فرض **سياسات دفع مسبق** على العملاء ذوي سجل إلغاءات مرتفع.
    - تحسين تجربة العملاء ذوي الطلبات الخاصة لأنهم الأكثر التزامًا.
    - استخدام نموذج التنبؤ لتحديد الحجوزات عالية المخاطر مبكرًا.
    """)

    st.markdown("---")
    st.markdown("---")
    st.subheader("📊 الرسوم البيانية الداعمة للتحليل")

    # 1) توزيع Lead Time
    st.write("### ⏳ توزيع Lead Time")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['lead_time'], bins=40, kde=True, color="#1A5276", ax=ax1)
    st.pyplot(fig1)

    # 2) توزيع ADR
    st.write("### 💵 توزيع ADR (متوسط سعر الغرفة)")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['adr'], bins=40, kde=True, color="#117864", ax=ax2)
    st.pyplot(fig2)

    # 3) نسبة الإلغاء حسب نوع الفندق
    st.write("### 🏨 نسبة الإلغاء حسب نوع الفندق")
    cancel_by_hotel = df.groupby('hotel')['is_canceled'].mean() * 100
    fig3, ax3 = plt.subplots()
    cancel_by_hotel.plot(kind='bar', color=['#1A5276', '#117864'], ax=ax3)
    ax3.set_ylabel("Cancellation Rate (%)")
    st.pyplot(fig3)

    # 4) الإلغاءات حسب الشهر
    st.write("### 📅 الإلغاءات حسب شهر الوصول")
    fig4, ax4 = plt.subplots()
    sns.countplot(x=df['arrival_date_month'], hue=df['is_canceled'], palette="Blues", ax=ax4)
    plt.xticks(rotation=45)
    st.pyplot(fig4)
    st.write("### ⏳ توزيع Lead Time (تفاعلي)")
    fig1 = px.histogram(df, x="lead_time", nbins=40, title="Lead Time Distribution",                         color_discrete_sequence=["#1A5276"])
    st.plotly_chart(fig1, use_container_width=True)
    st.write("### 💵 توزيع ADR (تفاعلي)")
    fig2 = px.histogram(df,x="adr",nbins=40,title="ADR Distribution",color_discrete_sequence=s    ["#117864"])
 
    st.plotly_chart(fig2, use_container_width=True)
    st.write("### 🏨 نسبة الإلغاء حسب نوع الفندق (تفاعلي)")
    cancel_by_hotel = df.groupby("hotel")["is_canceled"].mean().reset_index()
    cancel_by_hotel["is_canceled"] *= 100
    fig3 = px.bar(cancel_by_hotel, x="hotel", y="is_canceled", title="Cancellation Rate by Hotel",              color="hotel")
    st.plotly_chart(fig3, use_container_width=True)
    st.write("### 📅 الإلغاءات حسب شهر الوصول (تفاعلي)")
    fig4 = px.histogram(df, x="arrival_date_month", color="is_canceled", barmode="group",                    title="Cancellations by Month")
    st.plotly_chart(fig4, use_container_width=True)

    st.success("📄 صفحة Insights جاهزة الآن ويمكنك إضافتها إلى التقرير النهائي أو الـ PDF.")
# ============================================
# صفحة Executive Dashboard
# ============================================
if page == "📊 Executive Dashboard":

    st.markdown("""
    <div style="
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;">
    <h2 style="color:#1A5276;">📊 Executive Dashboard</h2>
    <p>لوحة تحكم تنفيذية تعرض أهم مؤشرات الأداء والاتجاهات.</p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    total_bookings = len(df)
    cancel_rate = df['is_canceled'].mean() * 100
    avg_adr = df['adr'].mean()
    avg_lead = df['lead_time'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("إجمالي الحجوزات", total_bookings)
    col2.metric("نسبة الإلغاء", f"{cancel_rate:.2f}%")
    col3.metric("متوسط ADR", f"{avg_adr:.2f}")
    col4.metric("متوسط Lead Time", f"{avg_lead:.1f} يوم")

    st.markdown("---")

    # Chart 1: Monthly Booking Trend
    st.write("### 📈 اتجاه الحجوزات حسب الشهر")
    monthly = df.groupby('arrival_date_month')['is_canceled'].count()
    fig1, ax1 = plt.subplots()
    monthly.plot(kind='line', marker='o', color="#1A5276", ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # Chart 2: Cancellation Rate by Month
    st.write("### ❌ نسبة الإلغاء حسب الشهر")
    cancel_month = df.groupby('arrival_date_month')['is_canceled'].mean() * 100
    fig2, ax2 = plt.subplots()
    cancel_month.plot(kind='bar', color="#C0392B", ax=ax2)
    st.pyplot(fig2)

    # Chart 3: Heatmap
    st.write("### 🔥 Heatmap: العلاقة بين المتغيرات")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_model.corr(), cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    # Chart 4: ADR vs Lead Time
    st.write("### 💵 ADR مقابل Lead Time")
    fig4, ax4 = plt.subplots()
    sns.scatterplot(x=df['lead_time'], y=df['adr'], hue=df['is_canceled'], palette="coolwarm", ax=ax4)
    st.pyplot(fig4)
