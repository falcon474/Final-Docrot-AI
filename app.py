import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import gdown
import os

# إعداد الصفحة
st.set_page_config(page_title="Doctor AI", page_icon="🩺")

# --- دالة التحميل من جوجل درايف ---
@st.cache_resource
def load_model():
    output_path = 'my_model.keras'
    if not os.path.exists(output_path):
        # 🔴🔴🔴 
        file_id = '11sSxpk1C_4x3edIdmRliO4wY7wSXx9rl' 
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)
    
    model = tf.keras.models.load_model(output_path)
    return model

# تحميل الموديل
with st.spinner('...'):
    try:
        model = load_model()
    except Exception as e:
        st.error("خطأ في الاتصال بالسحابة. تأكد من رابط Google Drive")
        st.stop()

# الواجهة
st.title("🩺 X-Ray Doctor AI")
st.write("AI to detect pneumonia and tuberculosis")

file = st.file_uploader("upload image here", type=["jpg", "png", "jpeg"])

def import_and_predict(image_data, model):
    size = (224, 224)
    image = image_data.convert('RGB')
    image = ImageOps.fit(image, size, Image.BILINEAR)
    img = np.asarray(image)
    img = img.astype(np.float32) / 255.0
    img_reshape = np.expand_dims(img, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

if file is not None:
    image = Image.open(file)
    st.image(image, width=300)
    
    if st.button('🔍 start detection'):
        predictions = import_and_predict(image, model)
        class_names = ['Normal', 'Pneumonia', 'Tuberculosis ']
        
        # النتائج
        idx = np.argmax(predictions)
        label = class_names[idx]
        score = np.max(predictions) * 100
        tb_prob = predictions[0][2] * 100
        
        st.divider()
        
        # المنطق الذكي
        if tb_prob > 5.0 and idx != 2:
            st.warning("⚠️ تحذير: اشتباه بوجود سل (TB) رغم أن النتيجة الأولية مختلفة!")
            st.error(f"التشخيص المقترح: {label}")
        elif idx == 0:
            st.success(f"✅ الحالة: {label} ({score:.1f}%)")
        else:
            st.error(f"⚠️ الحالة: {label} ({score:.1f}%)")
            
        # التفاصيل
        with st.expander("رؤية التفاصيل الرقمية"):
            st.write(f"Normal: {predictions[0][0]*100:.2f}%")
            st.write(f"Pneumonia: {predictions[0][1]*100:.2f}%")
            st.write(f"TB: {predictions[0][2]*100:.2f}%")
