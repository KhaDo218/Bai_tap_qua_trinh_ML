# Gom cụm Customer Chinook theo mức độ quan tâm Track và Album (K-Means)

**Sinh viên:** Đỗ Lê Quỳnh Kha · K244161790 · Học máy trong phân tích kinh doanh (MLBA)
**Kết quả trực quan:** mở `results/html/index.html` bằng trình duyệt (xem được cả khi offline)

## Cấu trúc thư mục

```
Chinook_Customer_Clustering/
├── Dataset/
│   └── Chinook_Sqlite.sqlite
├── Utils/                          # thư viện dùng chung
│   ├── ClusterUtil.py              # kết nối, tạo thuộc tính, Elbow + Kneedle, K-Means, vẽ, đặt tên cụm, gợi ý album
│   ├── ExportUtil.py               # xuất Excel (mỗi cụm 1 sheet) và trang HTML (plotly)
│   └── FileUtil.py                 # lưu / tải mô hình (pickle)
├── Clustering/                     # các bước máy học, chạy theo thứ tự
│   ├── data_exploration.py         # 1. khám phá dữ liệu
│   ├── CustomerCluster1.py         # 2. thực nghiệm 1: quan tâm TRACK  (GenreCount, AvgMinutes)
│   ├── CustomerCluster2.py         # 3. thực nghiệm 2: quan tâm ALBUM  (AlbumCount, AlbumPopularity)
│   ├── CustomerCluster3.py         # 4. thực nghiệm 3: TRACK + ALBUM   (3 thuộc tính, vẽ 3D)
│   ├── using_model.py              # 5. dùng mô hình trên console: cụm của khách, gợi ý album, khách mới
│   └── export_results.py           # 6. tải 3 mô hình đã lưu -> xuất results/html/index.html
├── Ui/                             # giao diện PyQt6
│   ├── ChinookMainWindow.ui        # thiết kế bằng Qt Designer
│   ├── ChinookMainWindow.py        # sinh bằng PyUIC
│   └── ChinookMainWindowEx.py      # xử lý sự kiện, tích hợp mô hình
├── Trained/                        # mô hình đã lưu (*_model.zip, *_scaler.zip)
├── results/
│   ├── excel/                      # CustomerCluster1_Track.xlsx, ...2_Album.xlsx, ...3_TrackAlbum.xlsx
│   ├── html/index.html             # trang kết quả: 5 tab (tổng quan, 3 thực nghiệm, tra cứu & gợi ý)
│   └── images/                     # ảnh chụp từ index.html để đưa vào báo cáo
├── main.py                         # chạy giao diện PyQt6
├── requirements.txt
└── README.md
```

Tất cả đường dẫn được tính từ vị trí file `Utils/ClusterUtil.py` (biến `BASE_DIR`, `DATABASE_PATH`,
`TRAINED_DIR`, `EXCEL_DIR`, `HTML_DIR`, `IMAGES_DIR`), nên chạy file ở thư mục nào cũng đúng.

## Cài thư viện

```
pip install -r requirements.txt
```

## Cách chạy (PyCharm: chuột phải từng file → Run)

1. `Clustering/data_exploration.py`
2. `Clustering/CustomerCluster1.py`, `CustomerCluster2.py`, `CustomerCluster3.py`
   → mỗi file: histogram, Elbow + Kneedle (tự chọn K), biểu đồ cụm, in WCSS / Silhouette,
   danh sách khách theo cụm, xuất `results/excel/*.xlsx`, lưu mô hình vào `Trained/`
3. `Clustering/using_model.py` (nhập CustomerId, ví dụ 12)
4. `Clustering/export_results.py` → tạo `results/html/index.html`
5. `main.py` → giao diện: Load Data → Elbow → Train Model → Save Model → Xem cụm & Gợi ý → Predict Cluster

Mỗi biểu đồ matplotlib hiện ra thì đóng cửa sổ để chương trình chạy tiếp. Biểu đồ 3D (plotly) mở trên trình duyệt.

## Chọn số cụm K: Elbow + Kneedle

Chuẩn hóa K và WCSS về [0, 1], đường thẳng nối điểm đầu và điểm cuối là y = 1 − x.
Độ gập của mỗi K là d = (1 − x) − y. K có d lớn nhất là điểm khuỷu tay; các K có d ≥ 90% giá trị lớn nhất tạo thành "khoảng elbow".

| Thực nghiệm | Khoảng elbow | K (điểm gập lớn nhất) | Silhouette |
|---|---|---|---|
| 1. GenreCount × AvgMinutes | 3 – 4 | 3 | 0.4338 |
| 2. AlbumCount × AlbumPopularity | 2 – 4 | 3 | 0.4135 |
| 3. GenreCount × AvgMinutes × AlbumCount | 3 – 4 | 4 | 0.3129 |

Bài làm lấy K tại điểm gập lớn nhất, không ưu tiên K lớn hơn trong khoảng elbow: ở thực nghiệm 1,
K = 4 làm Silhouette giảm từ 0.434 xuống 0.399 và tách nhỏ cụm mà không thêm ý nghĩa kinh doanh.

## Thuộc tính đề xuất

| Thuộc tính | Ý nghĩa | Quan tâm |
|---|---|---|
| GenreCount | Số thể loại khác nhau của các track đã mua. Nhiều = nghe đa dạng, ít = tập trung | Track |
| AvgMinutes | Thời lượng trung bình các track đã mua (phút). Cao = hay mua track video dài (TV Shows, Drama...) | Track |
| AlbumCount | Số album khác nhau đã mua track. Nhiều = mua rải rác, ít = gom vào ít album | Album |
| AlbumPopularity | Độ phổ biến trung bình của các album đã mua (độ phổ biến 1 album = số customer đã mua track của album đó) | Album |

Các thuộc tính được chuẩn hóa Min-Max về [0, 1] trước khi chạy K-Means, vì K-Means dùng khoảng cách Euclid.
Tên cụm được đặt tự động theo thuộc tính lệch nhiều nhất so với trung bình (z-score).
