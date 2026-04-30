# Tóm tắt quy trình chuyển đổi dữ liệu thô sang dữ liệu sạch

## Mục tiêu
Tạo ma trận đặc trưng theo ngày (huấn luyện/kiểm tra) từ tất cả các bảng CSV, bảo đảm không rò rỉ thông tin tương lai, sau đó làm sạch đặc trưng để huấn luyện mô hình.

## Nguồn dữ liệu đầu vào
- Phân tích: `data/analytical/sales.csv`, `data/analytical/sample_submission.csv`
- Dữ liệu gốc: `data/master/products.csv`, `data/master/customers.csv`, `data/master/geography.csv`, `data/master/promotions.csv`
- Vận hành: `data/operational/inventory.csv`, `data/operational/web_traffic.csv`
- Giao dịch: `data/transaction/orders.csv`, `data/transaction/order_items.csv`, `data/transaction/payments.csv`, `data/transaction/returns.csv`, `data/transaction/reviews.csv`, `data/transaction/shipments.csv`

## Quy trình chuyển đổi (rút gọn)
1. Tạo lịch ngày chuẩn từ 2012-07-04 đến 2024-07-01; thêm đặc trưng lịch (`day_sin/cos`, `day_of_week`, `month`, `weekend`) và sự kiện (Tết, ngày lễ). Chỉ dùng `Date` nên an toàn cho dự báo.
2. Xử lý dữ liệu gốc:
   - `products.csv`: biên lợi nhuận, nhóm giá, mã hóa `category/segment`.
   - `customers.csv` + `geography.csv`: tổng hợp theo `region` (đếm khách, tỉ lệ giới tính/tuổi, kênh acquisition).
3. Trích xuất mẫu lịch sử (chỉ đến 2022-12-31):
   - returns + order_items + products -> tỉ lệ trả hàng theo `category`.
   - promotions + orders + order_items + products -> hiệu quả khuyến mãi theo `category`.
   - web_traffic -> hồ sơ mùa vụ theo (ngày trong tuần, tháng).
   - orders + web_traffic -> tỉ lệ chuyển đổi theo kênh.
4. Đặc trưng vận hành:
   - web_traffic: tổng hợp theo ngày; giai đoạn dự báo dùng hồ sơ mùa vụ để suy ra `expected_sessions` và `traffic_uncertainty`.
   - inventory: ảnh chụp theo kỳ -> điền theo ngày, thêm `days_since_snapshot` và cờ stale.
5. Đặc trưng khuyến mãi:
   - Mở rộng promotions theo ngày (giai đoạn huấn luyện), tính `promo_active`, `promo_intensity`, hiệu ứng kéo dài và trọng số suy giảm.
   - Giai đoạn dự báo: dùng xác suất theo mùa hoặc đặt 0 (bảo thủ).
6. Ghép toàn bộ đặc trưng vào lịch ngày chuẩn bằng ghép trái; xử lý thiếu dữ liệu (điền theo giá trị gần nhất có giới hạn + cờ đã điền).
7. Tạo biến mục tiêu từ `sales.csv` (Revenue, COGS) cho huấn luyện; tách đặc trưng huấn luyện/kiểm tra.
8. Làm sạch đặc trưng: loại bỏ cột hằng, cột tương quan cao, cờ lịch dư thừa và biến chỉ mang xu hướng (`year`).

## Đầu ra
- Đặc trưng gốc: `output/train_features.csv`, `output/test_features.csv`, `output/train_target.csv`
- Đặc trưng đã làm sạch: `output/train_features_clean.csv`, `output/test_features_clean.csv`

## Scripts liên quan
- `scripts/process_data_pipeline.py`: tạo đặc trưng gốc từ tất cả bảng.
- `scripts/reduce_features.py`: loại bỏ đặc trưng dư thừa để ra dữ liệu sạch.
