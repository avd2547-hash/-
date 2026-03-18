import streamlit as st
import pandas as pd
from datetime import datetime
import os
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

# นำเข้าไลบรารีสำหรับ AI
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="แดชบอร์ดมือถือ (Big Data)", layout="wide", initial_sidebar_state="expanded")

# ตั้งค่าไทย - ฟอนต์สมัยใหม่
st.markdown("""
    <style>
    * { font-family: 'Segoe UI', 'Helvetica', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# รายชื่อจังหวัดในไทยพร้อมพิกัด (ละติจูด, ลองจิจูด) สำหรับแสดงบนแผนที่
PROVINCE_COORDS = {
    "กรุงเทพมหานคร": (13.7563, 100.5018),
    "เชียงใหม่": (18.7953, 98.9620),
    "ชลบุรี": (13.3611, 100.9847),
    "ภูเก็ต": (7.8804, 98.3923),
    "ขอนแก่น": (16.4322, 102.8236),
    "นครราชสีมา": (14.9799, 102.0978),
    "สงขลา": (7.1898, 100.5954),
    "ระยอง": (12.6814, 101.2816),
    "อุดรธานี": (17.4138, 102.7872),
    "หนองคาย": (17.8785, 102.7420),
    "นนทบุรี": (13.8591, 100.5217),
    "ปทุมธานี": (14.0208, 100.5250),
    "สมุทรปราการ": (13.5991, 100.5968),
    "อยุธยา": (14.3505, 100.5563),
    "สุราษฎร์ธานี": (9.1333, 99.3333)
}
THAI_PROVINCES = list(PROVINCE_COORDS.keys())

# โหลดข้อมูล (Cache Data)
@st.cache_data
def load_data():
    filename = 'simulated_smartphone_bigdata_100k.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        # แปลง Camera เป็นตัวเลข
        df['Camera'] = pd.to_numeric(df['Camera'], errors='coerce').fillna(0)
        
        # แปลง Release_Date เป็น Datetime
        if 'Release_Date' in df.columns:
            df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce')
            
        # จำลองข้อมูลจังหวัด หากยังไม่มีในไฟล์ CSV
        if 'Province' not in df.columns:
            np.random.seed(42)
            df['Province'] = np.random.choice(THAI_PROVINCES, size=len(df))
            
        return df
    return pd.DataFrame()

# โหลดข้อมูลเข้าแอป
df = load_data()

# ฟังก์ชัน AI โมเดล (Cache Resource)
@st.cache_resource
def train_price_prediction_model(data):
    if data.empty:
        return None, None
        
    features = ['Rating', 'Spec_score', 'Ram_GB', 'Battery_mAh', 
                'Display_inches', 'Inbuilt_Memory_GB', 'Fast_Charging_W']
    
    df_train = data.dropna(subset=features + ['Price', 'company']).copy()
    
    if df_train.empty:
        return None, None
    
    le = LabelEncoder()
    df_train['company_encoded'] = le.fit_transform(df_train['company'].astype(str))
    
    X = df_train[features + ['company_encoded']]
    y = df_train['Price']
    
    model = RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42)
    model.fit(X, y)
    
    return model, le

ai_model, label_encoder = train_price_prediction_model(df)

# Sidebar
with st.sidebar:
    st.title("🎛️ เมนูหลัก")
    page = st.radio("เลือกหน้า:", 
        ["📊 แดชบอร์ด", "📋 ข้อมูลทั้งหมด", "➕ เพิ่มข้อมูล", "🤖 ทำนายราคา (AI)", "📥 โหลดข้อมูล"])

# ==========================================
# หน้า: แดชบอร์ด
# ==========================================
if page == "📊 แดชบอร์ด":
    st.title("📱 แดชบอร์ดข้อมูลมือถือ (Big Data Analytics)")
    
    if not df.empty:
        # --- Metrics สเปคพื้นฐาน ---
        st.subheader("📌 ข้อมูลภาพรวมสเปคและราคา")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("📱 รุ่นทั้งหมด", f"{len(df):,}")
        with col2:
            st.metric("🏢 แบรนด์", df['company'].nunique())
        with col3:
            avg_price_thb = df['Price'].mean()
            st.metric("💰 ราคาเฉลี่ย", f"{avg_price_thb:,.0f} บาท")
        with col4:
            st.metric("⭐ คะแนนเฉลี่ย", f"{df['Rating'].mean():.2f}")
        with col5:
            st.metric("💾 แรมเฉลี่ย", f"{df['Ram_GB'].mean():.1f}GB")
        with col6:
            st.metric("🔋 แบตเฉลี่ย", f"{df['Battery_mAh'].mean():,.0f}mAh")
            
        # --- Metrics ข้อมูล E-commerce ---
        if 'Monthly_Sales_Volume' in df.columns:
            st.subheader("🛒 สถิติยอดขายและพฤติกรรมผู้บริโภค (E-Commerce)")
            ec_col1, ec_col2, ec_col3, ec_col4 = st.columns(4)
            with ec_col1:
                st.metric("📦 ยอดขายรวมทุกรุ่น (เครื่อง/เดือน)", f"{df['Monthly_Sales_Volume'].sum():,.0f}")
            with ec_col2:
                st.metric("👁️ ยอดเข้าชมสินค้ารวม (Views)", f"{df['Ecom_Page_Views'].sum():,.0f}")
            with ec_col3:
                avg_cart_rate = df['Add_to_Cart_Rate'].mean() * 100
                st.metric("🛒 อัตราหยิบลงตะกร้าเฉลี่ย", f"{avg_cart_rate:.2f}%")
            with ec_col4:
                top_selling_brand = df.groupby('company')['Monthly_Sales_Volume'].sum().idxmax()
                st.metric("🏆 แบรนด์ยอดขายสูงสุด", top_selling_brand)
        
        st.divider()

        # --- [ส่วนที่เพิ่มใหม่] แผนที่ประเทศไทย และ ยอดขายตามจังหวัด ---
        if 'Province' in df.columns and 'Monthly_Sales_Volume' in df.columns:
            st.subheader("🗺️ แผนที่แบรนด์ยอดฮิต และ พื้นที่ยอดนิยม")
            
            map_col, bar_col = st.columns([1.5, 1])
            
            with map_col:
                # หาว่าแต่ละจังหวัด แบรนด์ไหนมียอดขายรวมสูงสุด
                top_brand_prov = df.groupby(['Province', 'company'])['Monthly_Sales_Volume'].sum().reset_index()
                # เลือกเฉพาะแถวที่ยอดขายสูงสุดของแต่ละจังหวัด
                idx_max = top_brand_prov.groupby('Province')['Monthly_Sales_Volume'].idxmax()
                top_brand_prov = top_brand_prov.loc[idx_max].reset_index(drop=True)
                
                # เพิ่มพิกัด (Lat, Lon) ลงใน DataFrame
                top_brand_prov['Lat'] = top_brand_prov['Province'].map(lambda x: PROVINCE_COORDS.get(x, (13.7563, 100.5018))[0])
                top_brand_prov['Lon'] = top_brand_prov['Province'].map(lambda x: PROVINCE_COORDS.get(x, (13.7563, 100.5018))[1])
                
                # สร้างกราฟแผนที่ Mapbox
                fig_map = px.scatter_mapbox(top_brand_prov, lat="Lat", lon="Lon",
                                            hover_name="Province", 
                                            hover_data={"company": True, "Monthly_Sales_Volume": True, "Lat": False, "Lon": False},
                                            color="company", size="Monthly_Sales_Volume",
                                            color_discrete_sequence=px.colors.qualitative.Set1,
                                            zoom=4.5, center={"lat": 13.5, "lon": 100.5},
                                            mapbox_style="carto-positron",
                                            title="📍 แบรนด์ที่มียอดขายสูงสุดในแต่ละจังหวัด",
                                            labels={"company": "แบรนด์อันดับ 1", "Monthly_Sales_Volume": "ยอดขาย (เครื่อง)"})
                fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)
                
            with bar_col:
                # ดึงรายชื่อแบรนด์มาทำ Dropdown
                all_brands = sorted(df['company'].dropna().unique())
                selected_brand = st.selectbox("🎯 เลือกระบุแบรนด์เพื่อดูพื้นที่ขายดี:", ['ดูทุกแบรนด์รวมกัน'] + all_brands)
                
                # กรองข้อมูลตามที่เลือก
                if selected_brand == 'ดูทุกแบรนด์รวมกัน':
                    df_geo = df.copy()
                    chart_title = "ยอดขายรวมทุกแบรนด์"
                else:
                    df_geo = df[df['company'] == selected_brand]
                    chart_title = f"ยอดขายแบรนด์ {selected_brand}"
                    
                prov_sales = df_geo.groupby('Province')['Monthly_Sales_Volume'].sum().reset_index()
                prov_sales = prov_sales.sort_values(by='Monthly_Sales_Volume', ascending=True)
                
                fig_geo = px.bar(prov_sales, x='Monthly_Sales_Volume', y='Province', orientation='h',
                                 labels={"Province": "จังหวัด", "Monthly_Sales_Volume": "ยอดขายรวม (เครื่อง)"},
                                 title=chart_title, color='Monthly_Sales_Volume', color_continuous_scale='Blues')
                st.plotly_chart(fig_geo, use_container_width=True)

        st.divider()
        
        # --- กราฟ 1-2 ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💰 ราคาเฉลี่ยตามแบรนด์ Top 10 (บาท)")
            price_by_brand = df.groupby('company')['Price'].mean().sort_values(ascending=False).head(10)
            fig1 = px.bar(price_by_brand, labels={"value": "ราคา (บาท)", "company": "แบรนด์"})
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("⭐ คะแนนตามแบรนด์ Top 10")
            rating_by_brand = df.groupby('company')['Rating'].mean().sort_values(ascending=False).head(10)
            fig2 = px.bar(rating_by_brand, labels={"value": "คะแนน", "company": "แบรนด์"}, color_discrete_sequence=['#FF6B6B'])
            st.plotly_chart(fig2, use_container_width=True)
        
        # --- กราฟ 3-4 ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 การกระจายแรม (GB)")
            fig3 = px.histogram(df, x='Ram_GB', nbins=15, labels={"Ram_GB": "ขนาดแรม (GB)"})
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            st.subheader("🔋 การกระจายแบตเตอรี่ (mAh)")
            fig4 = px.histogram(df, x='Battery_mAh', nbins=15, labels={"Battery_mAh": "แบตเตอรี่ (mAh)"}, color_discrete_sequence=['#4ECDC4'])
            st.plotly_chart(fig4, use_container_width=True)
        
        # --- กราฟ 5-6 ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📱 ขนาดหน้าจอ (นิ้ว)")
            fig5 = px.histogram(df, x='Display_inches', nbins=15, labels={"Display_inches": "ขนาดหน้าจอ (นิ้ว)"}, color_discrete_sequence=['#95E1D3'])
            st.plotly_chart(fig5, use_container_width=True)
        
        with col2:
            st.subheader("🎬 แรม vs ราคา (บาท)")
            fig6 = px.scatter(df, x='Ram_GB', y='Price', color='Rating', size=df['Camera'].fillna(0)+1, labels={"Ram_GB": "แรม (GB)", "Price": "ราคา (บาท)", "Rating": "คะแนน"})
            st.plotly_chart(fig6, use_container_width=True)
        
        # --- กราฟ 7-8 ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📸 กล้องเฉลี่ยตามแบรนด์ Top 10")
            camera_by_brand = df.groupby('company')['Camera'].mean().sort_values(ascending=False).head(10)
            fig7 = px.bar(camera_by_brand, labels={"value": "กล้อง (MP)", "company": "แบรนด์"}, color_discrete_sequence=['#FFB347'])
            st.plotly_chart(fig7, use_container_width=True)
        
        with col2:
            st.subheader("🔌 ชาร์จเร็วเฉลี่ยตามแบรนด์ Top 10")
            charge_by_brand = df.groupby('company')['Fast_Charging_W'].mean().sort_values(ascending=False).head(10)
            fig8 = px.bar(charge_by_brand, labels={"value": "ชาร์จเร็ว (W)", "company": "แบรนด์"}, color_discrete_sequence=['#DDA15E'])
            st.plotly_chart(fig8, use_container_width=True)
        
        # --- กราฟ 9-10 ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💾 หน่วยความจำภายใน vs ราคา (บาท)")
            fig9 = px.scatter(df, x='Inbuilt_Memory_GB', y='Price', color='Rating', labels={"Inbuilt_Memory_GB": "หน่วยความจำ (GB)", "Price": "ราคา (บาท)"})
            st.plotly_chart(fig9, use_container_width=True)
        
        with col2:
            st.subheader("🎚️ คะแนนสเปคตามแบรนด์ Top 10")
            spec_by_brand = df.groupby('company')['Spec_score'].mean().sort_values(ascending=False).head(10)
            fig10 = px.bar(spec_by_brand, labels={"value": "คะแนนสเปค", "company": "แบรนด์"}, color_discrete_sequence=['#B8860B'])
            st.plotly_chart(fig10, use_container_width=True)
        
        # --- กราฟ 11-12 ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🤖 การกระจายตัวของเวอร์ชั่น Android")
            fig11 = px.histogram(df, x='Android_version', nbins=10, labels={"Android_version": "เวอร์ชั่น Android"}, color_discrete_sequence=['#20B2AA'])
            st.plotly_chart(fig11, use_container_width=True)
        
        with col2:
            st.subheader("⚡ Processor GHz vs ราคา (บาท)")
            fig12 = px.scatter(df, x='Processor_GHz', y='Price', color='Rating', labels={"Processor_GHz": "Processor (GHz)", "Price": "ราคา (บาท)"})
            st.plotly_chart(fig12, use_container_width=True)

        st.divider()

        # --- กราฟ 13-16 วิเคราะห์ Big Data E-commerce ---
        if 'Release_Date' in df.columns:
            st.subheader("📈 วิเคราะห์เทรนด์และพฤติกรรมผู้ใช้งาน (Big Data Insights)")
            
            # กราฟ 13: Time Series ยอดขายตามวันเปิดตัว
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📅 แนวโน้มยอดขายรวมของรุ่นที่เปิดตัวตามช่วงเวลา")
                df_time = df.dropna(subset=['Release_Date']).copy()
                sales_trend = df_time.groupby(df_time['Release_Date'].dt.to_period('M'))['Monthly_Sales_Volume'].sum().reset_index()
                sales_trend['Release_Date'] = sales_trend['Release_Date'].dt.to_timestamp()
                fig13 = px.line(sales_trend, x='Release_Date', y='Monthly_Sales_Volume', 
                                labels={"Release_Date": "เดือน/ปี ที่เปิดตัว", "Monthly_Sales_Volume": "ยอดขายรายเดือน (เครื่อง)"})
                fig13.update_traces(line_color='#8E44AD')
                st.plotly_chart(fig13, use_container_width=True)

            # กราฟ 14: Scatter ยอดวิว vs ยอดขาย
            with col2:
                st.subheader("👁️ ยอดเข้าชม (Views) vs ยอดขาย (Sales)")
                fig14 = px.scatter(df, x='Ecom_Page_Views', y='Monthly_Sales_Volume', color='company', 
                                   labels={"Ecom_Page_Views": "ยอดเข้าชมเพจ", "Monthly_Sales_Volume": "ยอดขาย"})
                st.plotly_chart(fig14, use_container_width=True)

            # กราฟ 15: Add to cart rate
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🛒 อัตราหยิบลงตะกร้า (Add to Cart %) ตามแบรนด์")
                cart_by_brand = df.groupby('company')['Add_to_Cart_Rate'].mean().sort_values(ascending=False).head(10) * 100
                fig15 = px.bar(cart_by_brand, labels={"value": "อัตราส่วน (%)", "company": "แบรนด์"}, color_discrete_sequence=['#2E86C1'])
                st.plotly_chart(fig15, use_container_width=True)

            # กราฟ 16: ส่วนแบ่งยอดขาย (Pie Chart)
            with col2:
                st.subheader("📦 สัดส่วนยอดขายรวมตามแบรนด์ (Market Share)")
                sales_by_brand = df.groupby('company')['Monthly_Sales_Volume'].sum().reset_index()
                sales_by_brand = sales_by_brand.sort_values(by='Monthly_Sales_Volume', ascending=False).head(10)
                fig16 = px.pie(sales_by_brand, values='Monthly_Sales_Volume', names='company', hole=0.4)
                st.plotly_chart(fig16, use_container_width=True)

    else:
        st.warning("ไม่มีข้อมูลสำหรับแสดงผล กรุณาโหลดข้อมูลก่อน")

# ==========================================
# หน้า: ข้อมูลทั้งหมด
# ==========================================
elif page == "📋 ข้อมูลทั้งหมด":
    st.title("📋 ข้อมูลมือถือทั้งหมด")
    
    if not df.empty:
        col1, col2 = st.columns(2)
        with col1:
            search = st.text_input("🔍 ค้นหารุ่นมือถือ:")
            if search:
                df_filtered = df[df['Name'].str.contains(search, case=False, na=False)]
                st.dataframe(df_filtered, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        
        with col2:
            st.subheader("🔗 กรองตามแบรนด์")
            brands = st.multiselect("เลือกแบรนด์:", df['company'].dropna().unique())
            if brands:
                df_brand = df[df['company'].isin(brands)]
                st.dataframe(df_brand, use_container_width=True)
    else:
        st.warning("ไม่มีข้อมูล กรุณาเพิ่มหรือโหลดข้อมูล")

# ==========================================
# หน้า: เพิ่มข้อมูล
# ==========================================
elif page == "➕ เพิ่มข้อมูล":
    st.title("➕ เพิ่มข้อมูลมือถือใหม่")
    
    with st.form("form_mobile"):
        st.subheader("📌 ข้อมูลสเปคทั่วไป")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("ชื่อรุ่น*")
            rating = st.number_input("คะแนน (0-5)*", 0.0, 5.0, 4.0)
            spec_score = st.number_input("คะแนนสเปค*", 0, 100)
            company = st.text_input("แบรนด์*")
            price = st.number_input("ราคา (บาท)*", 0)
        
        with col2:
            no_of_sim = st.text_input("ซิม")
            android = st.number_input("เวอร์ชั่น Android", 10, 15, 13)
            camera = st.number_input("กล้อง (MP)", 0)
            ram = st.number_input("แรม (GB)", 2, 24, 4)
            battery = st.number_input("แบตเตอรี่ (mAh)", 3000, 7000, 5000)
        
        col1, col2 = st.columns(2)
        with col1:
            screen = st.text_input("ความละเอียดหน้าจอ")
            processor = st.text_input("ชื่อ Processor")
        
        with col2:
            display_inches = st.number_input("ขนาดหน้าจอ (นิ้ว)", 5.0, 7.0, 6.5)
            processor_ghz = st.number_input("Processor (GHz)", 2.0, 3.5, 2.8)
        
        col1, col2 = st.columns(2)
        with col1:
            ext_memory = st.number_input("หน่วยความจำภายนอก (GB)", 0, 2048, 512)
            internal_memory = st.number_input("หน่วยความจำภายใน (GB)*", 32, 512, 128)
        
        with col2:
            fast_charge = st.number_input("ชาร์จเร็ว (W)", 0, 200, 33)
            
        st.markdown("---")
        st.subheader("🛒 ข้อมูลการขายและพฤติกรรมผู้บริโภค")
        
        col3, col4 = st.columns(2)
        with col3:
            release_date = st.date_input("วันที่เปิดตัว (Release Date)", datetime.today())
            monthly_sales = st.number_input("ยอดขายเฉลี่ยรายเดือน (เครื่อง)", 0, 1000000, 1000)
            province = st.selectbox("พื้นที่ที่ขายดีที่สุด (Province)", THAI_PROVINCES)
            
        with col4:
            page_views = st.number_input("ยอดเข้าชม (Ecom Page Views)", 0, 10000000, 15000)
            add_to_cart = st.number_input("อัตราหยิบลงตะกร้า (Add to Cart Rate 0.0-1.0)", 0.0, 1.0, 0.05, format="%.4f")
        
        submitted = st.form_submit_button("✅ บันทึกข้อมูล", use_container_width=True)
        
        if submitted:
            if name and company and price:
                new_row = {
                    'Name': name,
                    'Rating': rating,
                    'Spec_score': spec_score,
                    'No_of_sim': no_of_sim,
                    'Android_version': android,
                    'Price': price,
                    'company': company,
                    'Camera': camera,
                    'Screen_resolution': screen,
                    'Processor_name': processor,
                    'Ram_GB': ram,
                    'Battery_mAh': battery,
                    'Display_inches': display_inches,
                    'External_Memory_GB': ext_memory,
                    'Inbuilt_Memory_GB': internal_memory,
                    'Fast_Charging_W': fast_charge,
                    'Processor_GHz': processor_ghz,
                    'Release_Date': release_date.strftime('%Y-%m-%d'),
                    'Monthly_Sales_Volume': monthly_sales,
                    'Ecom_Page_Views': page_views,
                    'Add_to_Cart_Rate': add_to_cart,
                    'Province': province
                }
                
                if not df.empty:
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                else:
                    df = pd.DataFrame([new_row])
                    
                df.to_csv('simulated_smartphone_bigdata_100k.csv', index=False)
                st.success("✅ เพิ่มข้อมูลสำเร็จ!")
                
                st.cache_data.clear()
                st.cache_resource.clear() 
            else:
                st.error("❌ กรุณากรอกข้อมูล* ให้ครบถ้วน")

# ==========================================
# หน้า: ทำนายราคา (AI)
# ==========================================
elif page == "🤖 ทำนายราคา (AI)":
    st.title("🤖 AI ทำนายราคามือถือ (Price Predictor)")
    st.write("ระบุสเปคที่คุณสนใจ แล้วให้ AI ช่วยประเมินราคาขายที่เหมาะสม (เรียนรู้จากข้อมูลตลาดประเทศไทย)")
    
    if ai_model is None or label_encoder is None:
        st.warning("⚠️ ไม่พบข้อมูลที่สมบูรณ์เพียงพอสำหรับการให้ AI เรียนรู้ กรุณาตรวจสอบชุดข้อมูลของคุณ")
    else:
        existing_brands = sorted(df['company'].dropna().astype(str).unique()) if not df.empty else ["Samsung", "Apple"]
        
        with st.form("form_predict_price"):
            st.subheader("📝 กรอกข้อมูลสเปคฮาร์ดแวร์")
            
            col1, col2 = st.columns(2)
            with col1:
                p_company = st.selectbox("แบรนด์ (Company)", existing_brands)
                p_rating = st.number_input("คะแนนรีวิวคาดหวัง (Rating 0-5)", 0.0, 5.0, 4.0, step=0.1)
                p_spec_score = st.number_input("คะแนนสเปคโดยรวม (Spec_score 0-100)", 0, 100, 80)
                p_ram = st.number_input("แรม (Ram_GB)", 1.0, 32.0, 8.0, step=1.0)
                
            with col2:
                p_battery = st.number_input("ความจุแบตเตอรี่ (Battery_mAh)", 1000.0, 15000.0, 5000.0, step=100.0)
                p_display = st.number_input("ขนาดหน้าจอ (Display_inches)", 4.0, 10.0, 6.5, step=0.1)
                p_memory = st.number_input("ความจุภายใน (Inbuilt_Memory_GB)", 16.0, 2048.0, 128.0, step=16.0)
                p_fast_charge = st.number_input("ระบบชาร์จเร็ว (Fast_Charging_W)", 0.0, 300.0, 33.0, step=1.0)
            
            st.markdown("---")
            p_actual_price = st.number_input("💰 ราคาขายจริง (บาท) [ใส่เพื่อเปรียบเทียบกับ AI / ใส่ 0 หากไม่ต้องการ]", 0.0, 500000.0, 0.0, step=500.0)
            
            submitted_predict = st.form_submit_button("✨ ให้ AI ทำนายราคา", use_container_width=True)
            
        if submitted_predict:
            try:
                try:
                    comp_encoded = label_encoder.transform([p_company])[0]
                except ValueError:
                    comp_encoded = 0
                
                x_predict = pd.DataFrame([{
                    'Rating': p_rating,
                    'Spec_score': p_spec_score,
                    'Ram_GB': p_ram,
                    'Battery_mAh': p_battery,
                    'Display_inches': p_display,
                    'Inbuilt_Memory_GB': p_memory,
                    'Fast_Charging_W': p_fast_charge,
                    'company_encoded': comp_encoded
                }])
                
                predicted_price = ai_model.predict(x_predict)[0]
                
                st.divider()
                st.subheader("🎯 ผลลัพธ์การทำนาย")
                
                if p_actual_price > 0:
                    price_diff = predicted_price - p_actual_price
                    col_p1, col_p2, col_p3 = st.columns(3)
                    
                    with col_p1:
                        st.info("💰 ราคาขายจริง (ที่คุณกรอก)")
                        st.header(f"{p_actual_price:,.0f} บาท")
                        
                    with col_p2:
                        st.success("🤖 ราคา AI ประเมิน")
                        st.header(f"{predicted_price:,.0f} บาท")
                        
                    with col_p3:
                        if price_diff > 0:
                            st.warning("คุ้มค่ากว่าสเปค 🚀")
                            st.metric("ส่วนต่างราคา (บาท)", f"+ {abs(price_diff):,.0f} บาท", delta=f"ประเมินสูงกว่าจริง {abs(price_diff):,.0f} บาท", delta_color="normal")
                        else:
                            st.error("ราคาแรงเกินสเปค 💸")
                            st.metric("ส่วนต่างราคา (บาท)", f"- {abs(price_diff):,.0f} บาท", delta=f"ประเมินต่ำกว่าจริง {abs(price_diff):,.0f} บาท", delta_color="inverse")
                else:
                    st.success(f"📱 ด้วยสเปคนี้ AI ประเมินราคาขายที่เหมาะสมอยู่ที่:")
                    st.title(f"👉 {predicted_price:,.0f} บาท")
                    
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการประมวลผล - {e}")

# ==========================================
# หน้า: โหลดข้อมูล
# ==========================================
elif page == "📥 โหลดข้อมูล":
    st.title("📥 โหลดไฟล์ข้อมูล")
    
    if not df.empty:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 โหลด CSV", use_container_width=True):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 คลิกเพื่อโหลด",
                    data=csv,
                    file_name=f"bigdata_mobile_data_{datetime.now().strftime('%d%m%Y')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📥 โหลด Excel", use_container_width=True):
                excel_file = "temp.xlsx"
                if 'Release_Date' in df.columns:
                    df['Release_Date'] = df['Release_Date'].dt.tz_localize(None) 
                df.to_excel(excel_file, index=False)
                with open(excel_file, "rb") as f:
                    st.download_button(
                        label="📥 คลิกเพื่อโหลด",
                        data=f,
                        file_name=f"bigdata_mobile_data_{datetime.now().strftime('%d%m%Y')}.xlsx",
                        mime="application/vnd.ms-excel"
                    )
        
        st.divider()
        st.subheader("📊 สถิติไฟล์ปัจจุบัน")
        st.write(f"- จำนวนข้อมูล: {len(df):,} แถว")
        st.write(f"- จำนวนคอลัมน์: {len(df.columns)} คอลัมน์")
        st.write(f"- วันที่อัปเดตล่าสุด: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    else:
        st.warning("ไม่มีข้อมูลสำหรับให้ดาวน์โหลด")

st.divider()
st.caption("💡 แดชบอร์ดข้อมูลมือถือ & AI (Big Data Edition) | ทำด้วย Streamlit + Python 🚀")