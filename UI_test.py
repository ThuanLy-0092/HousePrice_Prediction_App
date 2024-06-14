import streamlit as st
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from datetime import datetime
from geopy.geocoders import Nominatim

st.markdown(
    """
    <style>
    #MainMenu img {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)
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

# Đọc dữ liệu
database = pd.read_csv("data_base.csv")

# Chuyển đổi cột 'Date Posted' sang datetime
database['Date Posted'] = pd.to_datetime(database['Date Posted'])

# Tìm ngày nhỏ nhất trong tập dữ liệu
min_date = database['Date Posted'].min()

# Thêm cột 'days_since_min' vào dữ liệu
database['days_since_min'] = (database['Date Posted'] - min_date).dt.days

# Đọc dữ liệu mô hình
data = pd.read_csv("clean_data.csv")

# Chuẩn bị dữ liệu đầu vào mô hình
X = data.drop(['Price'], axis=1)
y = data['Price']

# Chia dữ liệu
X_main, X_test, y_main, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=0.176, random_state=0)

# Huấn luyện mô hình
linear_regression = linear_model.LinearRegression()
linear_regression.fit(X_train, y_train)
st.image('logo.png', use_column_width=True)
# Xây dựng giao diện với Streamlit
st.title('Dự đoán giá nhà ở Hồ Chí Minh')
import streamlit as st

# Tiêu đề trên sidebar
# Tiêu đề trên sidebar
st.sidebar.title("Giới thiệu")

# Đoạn Markdown giới thiệu ứng dụng của bạn
st.sidebar.markdown("""
**Dự đoán giá nhà ở Hồ Chí Minh**

Đây là ứng dụng dự đoán giá nhà ở thành phố Hồ Chí Minh. Bạn có thể nhập vào các thông tin như diện tích, số tầng, ngày đăng, số phòng và địa chỉ để dự đoán giá nhà.

Ứng dụng sử dụng mô hình học máy để dự đoán, và cung cấp thông tin chi tiết về khoảng cách từ địa chỉ nhập vào tới trung tâm Quận 1. Ngoài ra, bạn cũng có thể xem một video hướng dẫn bên dưới.

Hãy thử nhập các thông tin của bạn vào và khám phá tính năng của ứng dụng!

--- 
                    
**Thông tin liên hệ**
""")
st.sidebar.image('Jack.jpg', width=200)

# Nhập các biến số từ người dùng
area = st.number_input('Diện Tích:', min_value=0.0, step=0.1)
floors = st.number_input('Số tầng:', min_value=0, step=1)
date_posted = st.date_input('Ngày đăng')
rooms = st.number_input('Số phòng', min_value=0)
address = st.text_input('Địa chỉ')

# Biến trạng thái để điều khiển quá trình hiển thị giao diện và dự đoán
show_prediction = False

# Nút bấm xác nhận
if st.button('Xác nhận'):
    # Chuyển đổi ngày đăng thành số ngày từ mốc min_date
    date_posted = pd.to_datetime(date_posted)
    days_since_min = (date_posted - min_date).days

    # Geocoding địa chỉ
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.geocode(address)
    if location:
        lon, lat = location.longitude, location.latitude
        distance_to_center = haversine(lon, lat, center_longitude, center_latitude)
        
        # Hiển thị thông tin xác nhận
        st.write(f"Diện tích: {area}")
        st.write(f"Số tầng: {floors}")
        st.write(f"Ngày đăng: {date_posted}")
        st.write(f"Số phòng: {rooms}")
        st.write(f"Địa chỉ: {address}")
    else:
        st.error('Không thể xác định tọa độ của địa chỉ đã nhập. Vui lòng kiểm tra lại.')

    input_data = pd.DataFrame({
        'Acreage': [area],
        'Floors': [floors],
        'days_since_min': [days_since_min],
        'Rooms': [rooms],
        'Distance_to_center': [distance_to_center]
    })

    # Dự đoán
    prediction = linear_regression.predict(input_data)[0]

    # Hiển thị dự đoán
    st.subheader(f'Dự đoán giá nhà: {prediction:.2f} tỷ')
    video_path = "idol.mp4"  # Đường dẫn đến file video trong thư mục của ứng dụng Streamlit
    st.video(video_path)
