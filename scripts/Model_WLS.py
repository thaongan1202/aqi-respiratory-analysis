import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

# 1. Đọc dữ liệu đã merge sẵn của nhóm
print("Đang đọc dữ liệu...")
df = pd.read_csv("../data/AQI_Respiratory_Disease_Final.csv")

# Chia dữ liệu theo từng bệnh
df_copd = df[df["Disease"] == "COPD"].copy()
df_asthma = df[df["Disease"] == "Current Asthma"].copy()

# Công thức mô hình tối ưu (Thêm Log_Population đại diện cho mức độ đô thị hóa/kinh tế)
formula = "Disease_AdjPrev ~ Median_AQI + Smoking_AdjPrev + Log_Population + C(Year)"

def build_optimal_model(data, disease_name):
    print(f"\n--- XÂY DỰNG MÔ HÌNH CHO: {disease_name} ---")
    
    # BƯỚC 1: Mô hình sơ bộ để tìm điểm bất thường (Outliers)
    model_prelim = smf.ols(formula, data=data).fit()
    
    # Tính Cook's Distance
    influence = model_prelim.get_influence()
    cooks_d = influence.cooks_distance[0]
    
    # Lọc bỏ Outliers (Ngưỡng phổ biến: 4 / N)
    threshold = 4 / len(data)
    data_clean = data[cooks_d < threshold].copy()
    print(f"Đã loại bỏ {len(data) - len(data_clean)} điểm bất thường (Outliers) bằng Cook's Distance.")
    
    # BƯỚC 2: Mô hình WLS (Trọng số theo Dân số) để xử lý nhiễu/hình phễu
    # Dùng Population làm trọng số (Khu vực đông dân thì sai số ít hơn)
    model_optimal = smf.wls(formula, data=data_clean, weights=data_clean["Population"]).fit(
        cov_type="cluster", cov_kwds={"groups": data_clean["CountyFIPS"]}
    )
    print(f"R-squared: {model_optimal.rsquared:.4f}")
    
    # BƯỚC 3: Vẽ biểu đồ Sai số (Residual Plot)
    plt.figure(figsize=(8, 5))
    plt.scatter(model_optimal.fittedvalues, model_optimal.resid, alpha=0.3, color='blue' if disease_name == "COPD" else 'orange')
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"Residual Plot - {disease_name}\n(Sai số phân tán ngẫu nhiên -> Mô hình tốt)")
    plt.xlabel("Fitted Values (Giá trị dự đoán)")
    plt.ylabel("Residuals (Sai số)")
    plt.tight_layout()
    plt.savefig(f"../images/Residual_Plot_{disease_name.replace(' ', '_')}.png")
    plt.close()
    print(f"Đã lưu biểu đồ sai số: Residual_Plot_{disease_name.replace(' ', '_')}.png")
    
    return model_optimal, data_clean

model_copd, df_copd_clean = build_optimal_model(df_copd, "COPD")
model_asthma, df_asthma_clean = build_optimal_model(df_asthma, "Current Asthma")

# BƯỚC 4: Tạo tập dữ liệu DỰ BÁO chuẩn cho Tableau
print("\n--- ĐANG TẠO DỮ LIỆU DỰ BÁO CHO TABLEAU ---")
def create_forecast(model, data_clean, disease_name):
    # Cho AQI chạy từ mức thấp nhất đến cao nhất của tập dữ liệu đã làm sạch
    aqi_range = np.linspace(data_clean["Median_AQI"].min(), data_clean["Median_AQI"].max(), 50)
    
    # Cố định các biến kiểm soát ở mức Trung bình / Phổ biến nhất
    input_du_bao = pd.DataFrame({
        "Median_AQI": aqi_range,
        "Smoking_AdjPrev": data_clean["Smoking_AdjPrev"].mean(),
        "Log_Population": data_clean["Log_Population"].mean(),
        "Year": data_clean["Year"].mode()[0]
    })
    
    # Dự báo và lấy khoảng tin cậy 95%
    forecast_results = model.get_prediction(input_du_bao).summary_frame(alpha=0.05)
    forecast_results["Median_AQI"] = aqi_range
    forecast_results["Disease"] = disease_name
    
    return forecast_results[["Disease", "Median_AQI", "mean", "mean_ci_lower", "mean_ci_upper"]]

forecast_copd = create_forecast(model_copd, df_copd_clean, "COPD")
forecast_asthma = create_forecast(model_asthma, df_asthma_clean, "Current Asthma")

final_forecast = pd.concat([forecast_copd, forecast_asthma], ignore_index=True)

# Đổi tên cột cho dễ hiểu trên Tableau
final_forecast.rename(columns={
    "mean": "Predicted_Disease_Rate",
    "mean_ci_lower": "CI_Lower",
    "mean_ci_upper": "CI_Upper"
}, inplace=True)

final_forecast.to_csv("../data/Tableau_Final_Forecast.csv", index=False)
print("ĐÃ LƯU FILE: Tableau_Final_Forecast.csv")

