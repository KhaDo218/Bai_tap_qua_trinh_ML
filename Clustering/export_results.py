# XUẤT KẾT QUẢ RA TRANG HTML (results/html/index.html)
# Dùng 3 mô hình đã lưu trong Trained (chạy CustomerCluster1.py, 2, 3 trước), không train lại
import html
import os

import pandas as pd
from sklearn.metrics import silhouette_score

from Chinook_Customer_Clustering.Utils.ClusterUtil import *
from Chinook_Customer_Clustering.Utils.ExportUtil import exportClustersToHtml
from Chinook_Customer_Clustering.Utils.FileUtil import FileUtil

conn = getConnect(DATABASE_PATH)
dfFeature = getCustomerFeatures(conn)
dfAlbum = getAlbumPurchaseDataset(conn)
dfPurchase = getPurchaseDataset(conn)

experiments = []
for exp in EXPERIMENTS:
    modelFile, scalerFile = getModelPaths(exp["file"])
    if not os.path.exists(modelFile) or not os.path.exists(scalerFile):
        raise SystemExit("Chưa có " + modelFile + " -> hãy chạy CustomerCluster1.py, 2, 3 trước")
    model = FileUtil.loadmodel(modelFile)
    scaler = FileUtil.loadmodel(scalerFile)
    columns = scaler["columns"]
    # 1. Chuẩn hóa bằng min, max đã lưu rồi dự đoán cụm cho 59 customer
    X = ((dfFeature[columns] - scaler["min"]) / (scaler["max"] - scaler["min"])).values
    df2 = dfFeature.copy()
    df2["cluster"] = model.predict(X)
    # 2. Elbow + Kneedle (để vẽ lại đồ thị chọn K)
    kRange, inertia, elbowZone, optimalK = findElbow(X)
    # 3. Bảng chi tiết customer + tên cụm
    dfDetail = getClusterDetail(conn, df2, columns)
    names = dict(zip(dfDetail["cluster"], dfDetail["ClusterName"]))
    # 4. Thống kê từng cụm, 5 thể loại và 5 album được mua nhiều nhất trong cụm
    dfAlbumCluster = dfAlbum.merge(dfDetail[["CustomerId", "cluster"]], on="CustomerId")
    rows, topGenres, topAlbums = [], {}, {}
    for c in sorted(dfDetail["cluster"].unique()):
        dfC = dfDetail[dfDetail["cluster"] == c]
        dfA = dfAlbumCluster[dfAlbumCluster["cluster"] == c]
        genreShare = (dfA["Genre"].value_counts(normalize=True) * 100).round(1)
        topGenres[c] = genreShare.head(5).rename_axis("Genre").reset_index(name="Tỷ lệ lượt mua (%)")
        topAlbums[c] = dfA.groupby(["Title", "Artist"])["CustomerId"].nunique() \
            .reset_index(name="Số khách") \
            .sort_values(["Số khách", "Title"], ascending=[False, True]).head(5)
        row = {"Cụm": f"Cluster {c}", "Tên cụm": names[c], "Số khách": len(dfC),
               "Tỷ lệ (%)": round(len(dfC) / len(dfDetail) * 100, 1)}
        for col in columns:
            row[f"{col} TB"] = round(dfC[col].mean(), 2)
        row["Chi tiêu TB (USD)"] = round(dfC["TotalSpent"].mean(), 2)
        row["Thể loại chính"] = ", ".join(f"{g} {p:.0f}%" for g, p in genreShare.head(3).items())
        rows.append(row)
    experiments.append({
        "name": exp["name"], "columns": columns, "df": dfDetail, "names": names,
        "k": model.n_clusters, "kRange": kRange.tolist(), "inertia": inertia.tolist(),
        "elbowZone": elbowZone.tolist(), "wcss": model.inertia_,
        "silhouette": silhouette_score(X, df2["cluster"]),
        "summary": pd.DataFrame(rows), "topGenres": topGenres, "topAlbums": topAlbums})
    print(exp["name"], "| K =", model.n_clusters, "| Kneedle K =", optimalK,
          "| Silhouette =", round(experiments[-1]["silhouette"], 4))

# 5. Bảng tra cứu: cụm của từng customer ở 3 mô hình + 5 album gợi ý (theo mô hình ALBUM)
dfLookup = getCustomerInfo(conn)[["CustomerId", "FullName", "Country"]].merge(dfFeature, on="CustomerId")
for i, e in enumerate(experiments, start=1):
    dfLookup = dfLookup.merge(e["df"][["CustomerId", "cluster"]].rename(columns={"cluster": f"Cụm TN{i}"}),
                              on="CustomerId")
dfAlbumModel = experiments[1]["df"]
recommendations = []
for customerId in dfLookup["CustomerId"]:
    albumCluster = dfAlbumModel[dfAlbumModel["CustomerId"] == customerId]["cluster"].values[0]
    sameClusterIds = dfAlbumModel[dfAlbumModel["cluster"] == albumCluster]["CustomerId"]
    dfRecommend = recommendAlbums(dfAlbum, sameClusterIds, customerId, 5)
    # bảng tra cứu cho phép HTML (escape=False) nên phải tự escape tên album / nghệ sĩ
    recommendations.append("<br>".join(f"{i}. {html.escape(t)} <i>({html.escape(a)})</i>" for i, (t, a)
                                       in enumerate(zip(dfRecommend["Title"], dfRecommend["Artist"]), start=1)))
for col in ["FullName", "Country"]:
    dfLookup[col] = dfLookup[col].map(html.escape)
dfLookup["5 album gợi ý"] = recommendations

# 6. Trang tổng quan
overview = {
    "cards": [("Khách hàng", len(dfFeature)),
              ("Lượt mua track", f"{len(dfPurchase):,}".replace(",", ".")),
              ("Album đã bán", dfPurchase["AlbumId"].nunique()),
              ("Thể loại đã bán", dfPurchase["GenreId"].nunique()),
              ("Mô hình K-Means", len(experiments))],
    "features": pd.DataFrame([
        ["GenreCount", "Track", "Số thể loại khác nhau đã mua", "Nghe đa dạng hay tập trung vài thể loại"],
        ["AvgMinutes", "Track", "Thời lượng trung bình (phút) các track đã mua", "Nghe bài hát ngắn hay mua video dài"],
        ["AlbumCount", "Album", "Số album khác nhau đã mua track", "Mua rải rác nhiều album hay gom vào ít album"],
        ["AlbumPopularity", "Album", "Số khách trung bình đã mua các album mà khách này mua",
         "Theo album phổ biến hay săn album ít người biết"]],
        columns=["Thuộc tính", "Quan tâm", "Cách tính", "Ý nghĩa"]),
    "footer": "Đỗ Lê Quỳnh Kha - K244161790 · Học máy trong phân tích kinh doanh · UEL"}

exportClustersToHtml(overview, experiments, dfLookup, os.path.join(HTML_DIR, "index.html"))
closeConnection(conn)
