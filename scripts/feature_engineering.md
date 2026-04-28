# Giải thích toàn bộ Feature Engineering — Tại sao lại làm vậy?

Tôi sẽ đi từ **nguyên lý cơ bản** trước, rồi mới giải thích từng bảng.

---

## Nguyên lý nền tảng: Mô hình học được gì?

Mô hình ML/DL **không hiểu thời gian**. Nó chỉ thấy một bảng số. Vì vậy nhiệm vụ của feature engineering là **dịch ngữ nghĩa kinh doanh và cấu trúc thời gian thành các con số** mà mô hình có thể học.

Câu hỏi cốt lõi khi thiết kế feature là:

> *"Thông tin này có giúp mô hình biết Revenue ngày mai sẽ cao hay thấp không?"*

Nếu có → làm. Nếu không → bỏ.

---

## Bảng 1: `sales.csv` — Lag & Rolling Features

### Tại sao dùng Lag?

Doanh thu hôm nay **không độc lập** với doanh thu hôm qua. Trong chuỗi thời gian, quá khứ gần nhất thường là dự báo tốt nhất cho tương lai gần. Đây gọi là tính **autocorrelation** (tự tương quan).

```
Lag-1:  Revenue hôm qua      → pattern ngắn hạn, momentum
Lag-7:  Revenue tuần trước   → pattern theo tuần (thứ 2 giống thứ 2)
Lag-14: Revenue 2 tuần trước → confirm pattern tuần
Lag-30: Revenue tháng trước  → pattern theo tháng
```

**Ví dụ cụ thể:** Nếu mỗi thứ 6 revenue đều cao hơn 30% so với ngày thường, mô hình cần thấy Revenue_lag7 (thứ 6 tuần trước) để học được điều đó. Nếu chỉ cho mô hình thấy ngày hôm nay là thứ 6, nó chưa đủ thông tin về *mức độ* cao bao nhiêu.

### Tại sao dùng Rolling Mean/Std?

- **Rolling mean** = xu hướng gần đây. Nếu 30 ngày gần nhất revenue trung bình tăng → mô hình biết đang trong uptrend
- **Rolling std** = mức độ biến động. Giai đoạn Tết có std rất cao (dao động lớn), mô hình cần biết để không bị ngạc nhiên

Rolling 7 ngày bắt xu hướng ngắn hạn, rolling 90 ngày bắt xu hướng mùa vụ.

### Tại sao dùng COGS ratio?

`Revenue / COGS` = proxy cho **gross margin**. Nếu ratio này giảm → có thể đang chạy khuyến mãi lớn hoặc mix sản phẩm thay đổi. Đây là tín hiệu về chất lượng doanh thu, giúp mô hình phân biệt "revenue cao vì bán nhiều" vs "revenue cao vì sản phẩm đắt tiền hơn".

---

## Bảng 2: `web_traffic.csv` — Leading Indicators

### Tại sao đây là bảng QUAN TRỌNG NHẤT sau sales?

Vì web traffic là **leading indicator** — nó xảy ra *trước* khi doanh thu phát sinh. Khách vào xem hàng hôm nay → mua hôm nay hoặc ngày mai. Đây là mối quan hệ nhân quả thực sự, không chỉ là tương quan.

```
Sessions          → bao nhiêu người có cơ hội mua
Unique visitors   → loại bỏ người vào nhiều lần, đo reach thực
Page views        → mức độ engagement (xem nhiều trang → mua nhiều hơn)
Bounce rate       → traffic chất lượng thấp hay cao
                    bounce cao → người vào rồi ra ngay → không mua
Avg session dur   → người dùng có đang đọc kỹ sản phẩm không?
```

### Tại sao lag traffic 1-7 ngày?

Vì hành vi mua hàng có **độ trễ (delay)**:
- Người vào xem hôm nay → so sánh → mua ngày mai (lag-1)
- Người xem cuối tuần → đợi lương → mua đầu tuần (lag-3 đến lag-7)

Nếu chỉ join traffic cùng ngày, mô hình bỏ lỡ toàn bộ hành vi "browsing trước, mua sau" này.

### Tại sao encode traffic_source?

Các kênh traffic có **conversion rate khác nhau**:
- Organic search → người tìm kiếm có chủ đích → mua nhiều hơn
- Social media → browsing thụ động → mua ít hơn
- Email campaign → đang chạy promo → spike revenue

Nếu một ngày traffic đến chủ yếu từ email campaign, mô hình nên dự báo revenue cao hơn bình thường dù tổng sessions bằng nhau.

---

## Bảng 3: `promotions.csv` — Event Features

### Tại sao không chỉ dùng lag mà cần promo flags?

Lag features học pattern **lặp lại thường xuyên**. Nhưng khuyến mãi là **sự kiện bất thường**, không lặp đều đặn. Nếu không báo cho mô hình biết "hôm nay có promo", nó sẽ không giải thích được spike doanh thu và coi đó là noise.

```
is_promo_active         → hôm nay có khuyến mãi không? (binary)
num_active_promos       → 1 promo vs 3 promo đồng thời → effect khác nhau
avg_discount_value      → giảm 10% vs giảm 50% → magnitude effect
days_to_next_promo      → anticipation effect: khách biết sắp có sale → trì hoãn mua
days_since_last_promo   → post-promo hangover: sau sale lớn, revenue thường giảm
```

### Tại sao cần "days to/since promo"?

Đây là hiện tượng thực tế trong ecommerce:

- **Trước sale:** Khách hoãn mua để chờ giảm giá → revenue giảm nhẹ vài ngày trước
- **Trong sale:** Revenue spike lên
- **Sau sale:** Revenue giảm mạnh vì nhu cầu đã được thỏa mãn trước đó (demand pull-forward)

Nếu không có feature này, mô hình sẽ thấy pattern giảm-tăng-giảm bất thường mà không hiểu tại sao.

---

## Bảng 4: `inventory.csv` — Supply Constraint

### Tại sao tồn kho ảnh hưởng đến doanh thu?

Đây là điểm nhiều người bỏ qua. Revenue không chỉ phụ thuộc vào **demand** (nhu cầu) mà còn phụ thuộc vào **supply** (khả năng đáp ứng).

```
stockout_days   → sản phẩm hết hàng bao nhiêu ngày trong tháng
                  → dù khách muốn mua, không có hàng → revenue bị cắt
fill_rate       → tỷ lệ đơn được đáp ứng đủ
                  → fill_rate thấp → revenue thực < revenue tiềm năng
reorder_flag    → cảnh báo sắp hết hàng → có thể stockout sắp xảy ra
```

**Ví dụ:** Tháng 11 có Black Friday, demand rất cao. Nhưng nếu fill_rate tháng đó chỉ đạt 60% (vì nhập hàng không kịp), revenue thực tế bị kìm hãm. Mô hình cần biết điều này để không overestimate revenue trong tương lai khi supply constraint vẫn còn.

### Tại sao aggregate theo tháng rồi join?

Vì inventory.csv chỉ có dữ liệu **cuối tháng** (monthly snapshot), không phải hàng ngày. Cách hợp lý là dùng inventory tháng T để làm feature cho tất cả các ngày trong tháng T+1 (tháng tiếp theo) — vì bạn chỉ biết tồn kho tháng này khi tháng kết thúc.

---

## Calendar Features — Tại sao quan trọng với thời trang Việt Nam?

### Tính mùa vụ (Seasonality)

Thời trang có tính mùa vụ rất cao:
- **Tháng 1-2:** Tết → spike lớn nhất năm (quần áo mới, tặng quà)
- **Tháng 3:** Sau Tết → sụt giảm
- **Tháng 8-9:** Back to school → tăng nhẹ
- **Tháng 11-12:** Black Friday + Giáng sinh + chuẩn bị Tết → tăng mạnh

Nếu không có calendar features, mô hình không phân biệt được tháng 1 và tháng 6.

### Tại sao `days_to_tet` quan trọng hơn chỉ biết "tháng 1"?

Tết Âm lịch **không cố định** theo Dương lịch. Năm nay Tết có thể ngày 29/1, năm sau là 17/2. Nếu chỉ dùng month=1 thì mô hình bị confuse vì pattern Tết trải dài khác nhau mỗi năm. `days_to_tet` chuẩn hóa điều này — "10 ngày trước Tết" luôn có pattern giống nhau dù năm nào.

```
days_to_tet < 0      → đã qua Tết
days_to_tet = 0      → ngày Tết
0 < days_to_tet < 7  → sắp Tết → mua sắm rầm rộ
```

### Tại sao `is_weekend`?

Hành vi mua hàng online ở Việt Nam cao hơn vào cuối tuần vì người dùng có thời gian rảnh. Đây là pattern ổn định, dễ học nhưng cần được encode tường minh.

---

## Bảng 5: `orders.csv` + `order_items.csv` — Tại sao để optional?

Về lý thuyết đây là nguồn thông tin rất giàu, nhưng có hai vấn đề:

**Vấn đề 1 — Data leakage tiềm ẩn:** Nếu bạn aggregate số đơn hàng theo ngày từ orders.csv, thực ra bạn đang tính lại revenue theo cách khác. Mô hình có thể overfit vào training data thay vì học pattern thực.

**Vấn đề 2 — Không có trong test period:** Bạn không có orders.csv cho giai đoạn 01/2023–07/2024. Mọi feature từ bảng này đều phải là lag (ít nhất 1 ngày), làm tăng độ phức tạp đáng kể.

Dùng nó khi muốn hiểu sâu hơn về **cấu trúc doanh thu** (ví dụ: average order value, tỷ lệ đơn có promo) nhưng không phải ưu tiên đầu tiên.

---

## Tổng kết: Bản đồ tư duy

```
Revenue ngày mai phụ thuộc vào:
│
├── Quá khứ gần (sales.csv lags)         → momentum, autocorrelation
├── Xu hướng (rolling stats)             → trend direction
├── Ý định mua (web_traffic lags)        → leading indicator
├── Kích thích mua (promotions)          → demand shock
├── Khả năng bán (inventory)             → supply constraint
└── Bối cảnh thời gian (calendar)        → seasonality, Tết effect
```

Mỗi nhóm feature giải thích **một phần khác nhau** của phương sai trong Revenue. Kết hợp tất cả lại mới cho mô hình bức tranh đầy đủ.