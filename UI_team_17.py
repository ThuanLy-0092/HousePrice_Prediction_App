import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

hide_elements_css = """
<style>
/* Ẩn biểu tượng GitHub và các lớp liên quan */
.css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK {
  display: none !important;
}

/* Ẩn menu chính (MainMenu) */
#MainMenu {
  visibility: hidden !important;
}

/* Ẩn footer */
footer {
  visibility: hidden !important;
}

/* Ẩn header */
header {
  visibility: hidden !important;
}
</style>
"""
st.set_page_config(page_title="Dự đoán giá nhà ở Hồ Chí Minh", page_icon="logo.png")
st.markdown(hide_elements_css, unsafe_allow_html=True)

st.image("Nen.webp")
# Hàm tính khoảng cách theo công thức Haversine
def haversine(lon1, lat1, lon2, lat2):
    # Chuyển đổi độ sang radian
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Công thức Haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Bán kính của Trái đất (trung bình) là 6371 km
    km = 6371 * c
    return km

# Tọa độ trung tâm Quận 1 (Longitude: 106.695833, Latitude: 10.776111)
center_longitude = 106.695833
center_latitude = 10.776111

# Đọc dữ liệu quận huyện từ GitHub
data_district = pd.read_csv("https://raw.githubusercontent.com/ThuanLy-0092/Test/main/Geocoded_district.csv")

# Đọc dữ liệu cơ sở dữ liệu và chuẩn bị mô hình
database = pd.read_csv("data_base.csv")
database['Date Posted'] = pd.to_datetime(database['Date Posted'])
min_date = database['Date Posted'].min()

# Load scaler và mô hình từ file
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('best_gb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# Xây dựng giao diện với Streamlit
st.sidebar.title("Giới thiệu")
st.sidebar.markdown("""
### **Dự đoán giá nhà ở Hồ Chí Minh**

### Đây là ứng dụng dự đoán giá nhà ở thành phố Hồ Chí Minh. Bạn có thể nhập vào các thông tin như diện tích, số tầng, ngày đăng, số phòng và địa chỉ để dự đoán giá nhà.

### Ứng dụng sử dụng mô hình học máy để dự đoán, và cung cấp thông tin chi tiết về khoảng cách từ địa chỉ nhập vào tới trung tâm Quận 1.

### Hãy thử nhập các thông tin của bạn vào và khám phá tính năng của ứng dụng!

### Nhóm tác giả:
- **Lý Vĩnh Thuận**
- **Nguyễn Nhựt Trường**
- **Từ Thức**

### Mô hình được sử dụng: XGBoost 
                    
""")

# Nhập các biến số từ người dùng
area = st.number_input('Diện Tích:', min_value=0.0, step=0.1)
floors = st.number_input('Số tầng:', min_value=0, step=1)
date_posted = st.date_input('Ngày mua/bán')
rooms = st.number_input('Số phòng', min_value=0)
amenities_rating = st.slider('Mức độ tiện nghi:', min_value=0, max_value=6, value=2)

# Widget cho lựa chọn quận
district_options = data_district['district'].unique()
selected_district = st.selectbox('Chọn quận:', district_options)

# Lọc dữ liệu để lấy ra longitude và latitude của quận được chọn
filtered_data = data_district[data_district['district'] == selected_district]

if not filtered_data.empty:
    # Lấy ra longitude và latitude của quận được chọn
    selected_longitude = filtered_data.iloc[0]['Longitude']
    selected_latitude = filtered_data.iloc[0]['Latitude']

    # Hiển thị thông tin về vị trí (Longitude và Latitude)
    st.write(f"Thông tin về vị trí của {selected_district}:")
    st.write(f"Longitude: {selected_longitude}")
    st.write(f"Latitude: {selected_latitude}")

    # Tính khoảng cách từ vị trí được chọn tới trung tâm Quận 1
    center_longitude = 106.695833  # Tọa độ trung tâm Quận 1
    center_latitude = 10.776111
    distance_to_center = haversine(selected_longitude, selected_latitude, center_longitude, center_latitude)

# Biến trạng thái để điều khiển quá trình hiển thị giao diện và dự đoán
show_prediction = False

# Nút bấm xác nhận
if st.button('Xác nhận'):
    # Chuyển đổi ngày đăng thành số ngày từ mốc min_date
    date_posted = pd.to_datetime(date_posted)
    days_since_min = (date_posted - min_date).days

    # Hiển thị thông tin xác nhận
    st.write(f"Diện tích: {area}m2")
    st.write(f"Số tầng: {floors}")
    st.write(f"Ngày đăng: {date_posted}")
    st.write(f"Số phòng: {rooms}")
    st.write(f"Mức độ tiện nghi: {amenities_rating}")

    # Tính toán khoảng cách tới trung tâm Quận 1 nếu có dữ liệu quận được chọn
    if not filtered_data.empty:
        st.write(f"Khoảng cách tới trung tâm Quận 1: {distance_to_center} km")

    # Tạo DataFrame từ dữ liệu đầu vào
    input_data = pd.DataFrame({
        'Acreage': [area],
        'Amenities_rating': [amenities_rating],
        'Floors': [floors],
        'days_since_min': [days_since_min],
        'Rooms': [rooms],
        'Distance_to_center': [distance_to_center],
    })

    # Chuẩn hóa dữ liệu đầu vào
    input_data_scaled = scaler.transform(input_data)

    # Dự đoán sử dụng mô hình XGBoost đã tải
    prediction = xgb_model.predict(input_data_scaled)[0]

    # Hiển thị dự đoán trên giao diện
    st.subheader(f'Dự đoán giá nhà: {prediction:.2f} tỷ')
