# E-commerce Fashion Analytics & Sales Forecasting

## Tổng Quan Dự Án

Dự án này phân tích dữ liệu thương mại điện tử trong ngành thời trang nhằm hiểu rõ các yếu tố ảnh hưởng đến doanh thu, lợi nhuận, hành vi khách hàng, hiệu quả khuyến mãi, vận hành logistics, tỷ lệ hoàn trả, web traffic và mức độ hài lòng của khách hàng.

Pipeline của nhóm gồm ba phần chính:

1. **Data Integration**: merge các bảng dữ liệu gốc thành các dataframe phân tích theo từng trục kinh doanh.
2. **Exploratory Data Analysis (EDA)**: phân tích insight bằng Jupyter Notebook.
3. **Sales Forecasting Model**: xây dựng mô hình dự báo `Revenue` và `COGS` cho giai đoạn test dựa trên dữ liệu chuỗi thời gian.

Mục tiêu cuối cùng là tạo ra insight kinh doanh có thể hành động được và xây dựng file submission dự báo doanh thu theo đúng định dạng yêu cầu.

---

## Cấu Trúc Dự Án

```text

## Cấu Trúc Dự Án

```text
├── EDA/                                  # Jupyter notebooks cho phân tích EDA
│   ├── EDA_inventory_returns.ipynb        # EDA về inventory và returns
│   ├── datathon_df1_eda.py                # EDA cho df_1: Revenue & Profit
│   ├── df_6_eda.ipynb                     # EDA cho df_6: Web Traffic → Conversion
│   └── df_7_eda (1).ipynb                 # EDA cho df_7: Reviews & Satisfaction
│
├── Merge_data_code/                      # Scripts merge dữ liệu theo từng trục phân tích
│   ├── merge_3.py                         # Tạo df_3: Promotion & Pricing
│   ├── merge_4.py                         # Tạo df_4: Delivery / Operations analysis
│   ├── merge_4_2.py                       # Tạo df_4_2: Inventory analysis
│   ├── merge_5.py                         # Tạo df_5: Product & Return
│   ├── merge_6_channel.py                 # Tạo df_6_channel: Conversion theo channel
│   ├── merge_6_daily.py                   # Tạo df_6_daily: Conversion theo ngày
│   ├── merge_7.py                         # Tạo df_7: Reviews & Satisfaction
│   ├── new_merge_df_1.py                  # Tạo df_1: Revenue & Profit
│   └── new_merge_df_2                     # Tạo df_2: Customer Behavior
│
├── Model/                                # Script huấn luyện mô hình forecasting
│   ├── submission_final.csv
│   ├── time_series_cv_metrics.csv
│   ├── time_series_cv_predictions.csv
│   ├── lgbm_feature_importance_gain.csv
│   ├── shap_feature_importance.csv
│
├── datathon_mcas.py                       # File Python chính / utility script của project
└── README.md
```text

Mô Tả Dữ Liệu

Dự án sử dụng nhiều bảng dữ liệu liên quan đến hoạt động thương mại điện tử:

File	Vai trò	Mô tả
products.csv	Master	Thông tin sản phẩm, category, segment, giá vốn
customers.csv	Master	Thông tin khách hàng và acquisition channel
promotions.csv	Master	Thông tin chương trình khuyến mãi
geography.csv	Master	Thông tin vùng, thành phố, zip code
orders.csv	Transaction	Thông tin đơn hàng
order_items.csv	Transaction	Chi tiết sản phẩm trong từng đơn
payments.csv	Transaction	Thông tin thanh toán
shipments.csv	Transaction	Thông tin vận chuyển
returns.csv	Transaction	Thông tin hoàn trả
reviews.csv	Transaction	Đánh giá của khách hàng
inventory.csv	Operational	Dữ liệu tồn kho
web_traffic.csv	Operational	Dữ liệu traffic website
sales.csv	Analytical	Dữ liệu chuỗi thời gian để train mô hình
sample_submission.csv	Analytical	File mẫu submission
Quy Trình Phân Tích

Pipeline phân tích được chia thành 7 trục chính.

Trục 1: Revenue & Profit — df_1
Business Questions
Doanh thu đến từ đâu?
Theo category
Theo segment
Theo region
Theo customer segment
Theo acquisition channel
Profitability có vấn đề không?
Sản phẩm nào bán nhiều nhưng biên lợi nhuận thấp?
Sản phẩm nào có margin cao nhưng bán kém?
Merge Logic
df_1 = order_items.merge(products, on="product_id", how="left")
df_1 = df_1.merge(orders, on="order_id", how="left")
df_1 = df_1.merge(customers, on="customer_id", how="left")
df_1 = df_1.merge(geography, left_on="zip", right_on="zip", how="left")
Key Metrics
df_1["revenue"] = df_1["price"] * df_1["quantity"]
df_1["profit"] = (df_1["price"] - df_1["cogs"]) * df_1["quantity"]
df_1["margin"] = (df_1["price"] - df_1["cogs"]) / df_1["price"]
Insight Direction
Segment nào có revenue cao nhưng margin thấp.
Region nào đóng góp doanh thu lớn nhất.
Acquisition channel nào đem lại nhóm khách hàng có giá trị cao.
Sản phẩm nào cần điều chỉnh giá hoặc discount.


Trục 2: Customer Behavior — df_2
Business Questions
Khách hàng nào là core customers?
Mỗi khách hàng đặt bao nhiêu đơn?
Tổng spending theo từng khách hàng là bao nhiêu?
Khoảng cách giữa hai lần mua hàng là bao lâu?
Acquisition channel nào đem lại khách hàng chất lượng nhất?
Merge Logic
df_2 = orders.merge(customers, on="customer_id", how="left")
df_2 = df_2.merge(order_items, on="order_id", how="left")
Key Metrics
customer_summary = df_2.groupby("customer_id").agg(
    num_orders=("order_id", "nunique"),
    total_spending=("price", "sum"),
    first_order_date=("order_date", "min"),
    last_order_date=("order_date", "max")
).reset_index()
Insight Direction
Channel có nhiều user nhưng spending thấp → low-quality acquisition.
Channel có ít user nhưng spending cao → high-value acquisition.
Nhóm khách hàng có frequency cao nên được retarget.
Nhóm khách hàng có spending cao nên được ưu tiên trong loyalty program.


Trục 3: Promotion & Pricing — df_3
Business Questions
Khuyến mãi có thực sự hiệu quả không?
Đơn có promo có revenue cao hơn đơn không có promo không?
Promo nào tốt nhất theo category, discount value hoặc promo type?
Discount cao có làm tăng volume hay chỉ làm giảm margin?
Merge Logic
df_3 = order_items.merge(promotions, on="promo_id", how="left")
df_3 = df_3.merge(products, on="product_id", how="left")
df_3 = df_3.merge(orders, on="order_id", how="left")
Key Metrics
df_3["has_promo"] = df_3["promo_id"].notna().astype(int)
df_3["revenue"] = df_3["price"] * df_3["quantity"]
df_3["profit"] = (df_3["price"] - df_3["cogs"]) * df_3["quantity"]
df_3["margin"] = (df_3["price"] - df_3["cogs"]) / df_3["price"]
Insight Direction
Promo nào tăng revenue nhưng làm giảm margin.
Discount level nào hiệu quả nhất.
Stackable promo có gây lỗ không.
Category nào phản ứng tốt nhất với promotion.


Trục 4: Operations — Logistics & Inventory
Trục này được chia thành hai dataframe:

df_4: Delivery analysis
df_4_2: Inventory analysis
4.1 Delivery Analysis — df_4
Business Questions
Delivery time có ảnh hưởng đến rating không?
Delivery chậm có làm tăng return rate không?
Khu vực nào có vấn đề logistics?
Merge Logic
df_4 = orders.merge(shipments, on="order_id", how="left")
df_4 = df_4.merge(reviews, on="order_id", how="left")
df_4 = df_4.merge(returns, on="order_id", how="left")
Key Metrics
df_4["ship_date"] = pd.to_datetime(df_4["ship_date"])
df_4["delivery_date"] = pd.to_datetime(df_4["delivery_date"])
df_4["delivery_time"] = (df_4["delivery_date"] - df_4["ship_date"]).dt.days
df_4["is_returned"] = df_4["return_id"].notna().astype(int)
Insight Direction
Delivery time tăng có thể làm rating giảm.
Delivery chậm có thể liên quan đến tỷ lệ return cao hơn.
Một số vùng có thể cần tối ưu logistics.
4.2 Inventory Analysis — df_4_2
Business Questions
Sản phẩm nào có nguy cơ stockout?
Sản phẩm nào bị overstock?
Tồn kho có ảnh hưởng đến doanh thu tiềm năng không?
Merge Logic
df_4_2 = inventory.merge(products, on="product_id", how="left")
Insight Direction
Stockout có thể làm mất doanh thu tiềm năng.
Overstock gây lãng phí chi phí lưu kho.
Inventory planning nên được cải thiện theo demand pattern.


Trục 5: Product & Return — df_5
Business Questions
Tại sao khách hàng trả hàng?
Category nào bị trả nhiều nhất?
Size hoặc color nào có tỷ lệ return cao?
Return reason phổ biến nhất là gì?
Merge Logic
df_5 = returns.merge(order_items, on="order_item_id", how="left")
df_5 = df_5.merge(products, on="product_id", how="left")
Key Metrics
return_summary = df_5.groupby(["category", "return_reason"]).agg(
    num_returns=("return_id", "count")
).reset_index()
Insight Direction
Size có return rate cao → cần cải thiện sizing guide.
Defective products → cần kiểm soát quality.
Một số category có vấn đề về expectation mismatch.


Trục 6: Web Traffic → Conversion
Trục này gồm hai dataframe:

df_6_daily: phân tích conversion theo ngày
df_6_channel: phân tích conversion theo channel
6.1 Daily Conversion — df_6_daily
Business Questions
Traffic tăng có dẫn đến sales tăng không?
Ngày nào có traffic cao nhưng conversion thấp?
Funnel có vấn đề ở giai đoạn nào?
Merge Logic
orders_daily = orders.groupby("order_date").agg(
    num_orders=("order_id", "nunique")
).reset_index()

traffic_daily = web_traffic.groupby("date").agg(
    total_sessions=("sessions", "sum"),
    total_visitors=("visitors", "sum")
).reset_index()

df_6_daily = traffic_daily.merge(
    orders_daily,
    left_on="date",
    right_on="order_date",
    how="left"
)
Key Metrics
df_6_daily["conversion_rate"] = df_6_daily["num_orders"] / df_6_daily["total_sessions"]
6.2 Channel Conversion — df_6_channel
Business Questions
Channel nào đem lại nhiều traffic nhất?
Channel nào có conversion tốt nhất?
Channel nào nhiều traffic nhưng low conversion?
Merge Logic
traffic_channel = web_traffic.groupby(["date", "channel"]).agg(
    sessions=("sessions", "sum"),
    visitors=("visitors", "sum")
).reset_index()

orders_channel = orders.merge(customers, on="customer_id", how="left")

orders_channel = orders_channel.groupby(["order_date", "acquisition_channel"]).agg(
    num_orders=("order_id", "nunique")
).reset_index()

df_6_channel = traffic_channel.merge(
    orders_channel,
    left_on=["date", "channel"],
    right_on=["order_date", "acquisition_channel"],
    how="left"
)
Insight Direction
Traffic cao nhưng conversion thấp → UX hoặc landing page có vấn đề.
Channel conversion cao → nên tăng marketing budget.
Channel nhiều visitors nhưng ít orders → cần tối ưu funnel.


Trục 7: Reviews & Satisfaction — df_7
Business Questions
Điều gì ảnh hưởng đến rating?
Delivery time có làm rating giảm không?
Category nào có review thấp?
Sản phẩm nào cần cải thiện chất lượng?
Merge Logic
df_7 = reviews.merge(orders, on="order_id", how="left")
df_7 = df_7.merge(shipments, on="order_id", how="left")
df_7 = df_7.merge(order_items, on="order_id", how="left")
df_7 = df_7.merge(products, on="product_id", how="left")
Key Metrics
df_7["delivery_time"] = (
    pd.to_datetime(df_7["delivery_date"]) - pd.to_datetime(df_7["ship_date"])
).dt.days
Insight Direction
Delivery time dài hơn có thể liên quan đến rating thấp.
Một số category có satisfaction thấp hơn trung bình.
Rating thấp có thể xuất phát từ logistics hoặc product quality.



Sau phần EDA, nhóm xây dựng mô hình dự báo chuỗi thời gian cho hai biến:

Revenue
COGS

File model chính:

src/train_ultimate_final.py

Mô hình sử dụng ensemble giữa hai thành phần:

Hybrid Linear Trend + LightGBM
Linear Regression học long-term trend trên log1p(target).
LightGBM học phần residual sau khi loại bỏ trend.
Pure LightGBM
LightGBM học trực tiếp log1p(target) từ calendar features và target encoding features.

Final prediction được tính bằng weighted ensemble:

final_prediction = 0.70 * hybrid_prediction + 0.30 * pure_lgbm_prediction

File model cũng có Time-Series Cross-Validation, loại bỏ một số giai đoạn COVID khỏi training, tạo feature importance, SHAP plots nếu package shap được cài đặt, và xuất file submission_final.csv.

Feature Engineering Cho Forecasting

Các feature chính bao gồm:

Calendar Features
month
day
dayofweek
quarter
is_weekend
is_month_start
is_month_end
is_double_day
is_mid_month
Holiday & Tết Features
days_to_tet
days_after_tet
is_tet_week
is_holiday
Target Encoding Features
month_mean_rev
dow_mean_rev
month_mean_cogs
dow_mean_cogs

Target encoding được tính từ training data hoặc fold training trong Cross-Validation để tránh data leakage.

Cài Đặt Môi Trường
1. Clone repository
git clone <repository-url>
cd <repository-name>
2. Tạo virtual environment
python -m venv venv

Kích hoạt môi trường:

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
3. Cài đặt thư viện
pip install -r requirements.txt

Nếu chưa có requirements.txt, có thể cài các package chính:

pip install pandas numpy scikit-learn lightgbm matplotlib shap jupyter
Cách Chạy Lại Toàn Bộ Pipeline
Bước 1: Đặt dữ liệu vào đúng folder

Đảm bảo folder dữ liệu có cấu trúc như sau:

datathon-2026-round-1/
├── sales.csv
├── sample_submission.csv
├── orders.csv
├── order_items.csv
├── products.csv
├── customers.csv
├── geography.csv
├── promotions.csv
├── shipments.csv
├── returns.csv
├── reviews.csv
├── inventory.csv
└── web_traffic.csv

Trong file model hiện tại, đường dẫn được cấu hình như sau:

DATA_DIR = PROJECT_ROOT / "datathon-2026-round-1"
OUTPUT_DIR = PROJECT_ROOT / "datathon-output"

Vì vậy, folder datathon-2026-round-1 cần nằm cùng cấp với file chạy model hoặc cùng cấp với project root mà script nhận diện.

Bước 2: Chạy notebook merge data

Mở Jupyter Notebook:

jupyter notebook

Sau đó chạy notebook merge:

notebooks/01_merge_data.ipynb

Notebook này tạo các dataframe phân tích:

df_1       Revenue & Profit
df_2       Customer Behavior
df_3       Promotion & Pricing
df_4       Delivery Analysis
df_4_2     Inventory Analysis
df_5       Product & Return
df_6_daily Web Traffic Daily Conversion
df_6_channel Web Traffic Channel Conversion
df_7       Reviews & Satisfaction

Bước 3: Chạy các notebook EDA

Mỗi notebook tập trung vào một nhóm business question riêng và tạo insight tương ứng.

Bước 4: Chạy forecasting model

Chạy file model:

python src/train_ultimate_final.py

Nếu file model đang nằm trực tiếp ở root folder, chạy:

python train_ultimate_final.py
Output Sau Khi Chạy Model

Sau khi chạy thành công, folder datathon-output/ sẽ được tạo tự động.

Các output chính gồm:

File	Mô tả
submission_final.csv	File submission cuối cùng gồm Date, Revenue, COGS
time_series_cv_metrics.csv	Kết quả Time-Series Cross-Validation
time_series_cv_predictions.csv	Dự đoán từng fold validation
lgbm_feature_importance_gain.csv	Feature importance theo LightGBM gain
shap_feature_importance.csv	SHAP feature importance nếu cài package shap
figures/	SHAP beeswarm plots và bar plots
Định Dạng Submission

File submission cuối cùng có format:

Date,Revenue,COGS
2023-01-01,2665507.2,2518885.15
2023-01-02,1280007.89,1136463.0
2023-01-03,1015899.51,822721.12

File được lưu tại:
datathon-output/submission_final.csv

Lưu Ý Khi Tái Lập Kết Quả
Cần đảm bảo đúng tên folder dữ liệu:
datathon-2026-round-1/
Cần đảm bảo có hai file bắt buộc cho modeling:
sales.csv
sample_submission.csv
Nếu gặp lỗi thiếu package shap, model vẫn chạy được. Khi đó script sẽ bỏ qua SHAP plots và vẫn xuất LightGBM feature importance.
Nếu thay đổi vị trí file model, cần kiểm tra lại biến:
PROJECT_ROOT
DATA_DIR
OUTPUT_DIR
Dữ liệu ngày tháng cần được parse đúng bằng pd.to_datetime() để tránh sai lệch timeline.
