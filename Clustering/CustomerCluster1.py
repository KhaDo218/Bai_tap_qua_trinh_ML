# THỰC NGHIỆM 1: Gom cụm Customer theo mức độ quan tâm tới TRACK
# Thuộc tính: GenreCount (số thể loại đã mua), AvgMinutes (thời lượng trung bình track đã mua)
import os

from sklearn.metrics import silhouette_score

from Chinook_Customer_Clustering.Utils.ClusterUtil import *
from Chinook_Customer_Clustering.Utils.ExportUtil import exportClustersToExcel
from Chinook_Customer_Clustering.Utils.FileUtil import FileUtil
conn = getConnect(DATABASE_PATH)
df2 = getCustomerFeatures(conn)
print(df2)
print(df2.head())
print(df2.describe())
columns = ["GenreCount", "AvgMinutes"]
showHistogram(df2, columns)
# Chuẩn hóa Min-Max trước khi gom cụm
dfScaled, minValues, maxValues = minMaxScale(df2, columns)
print(dfScaled.head())
# Elbow + Kneedle tự tìm số cụm (điểm gập lớn nhất)
cluster = elbowMethod(dfScaled, columns)
print("Số cụm được chọn: K =", cluster)
X = dfScaled.loc[:, columns].values
colors = ["red", "green", "blue", "purple", "black", "pink", "orange"]
y_kmeans, centroids, labels, model = runKMeans(X, cluster)
print(y_kmeans)
print(centroids)
print(labels)
df2["cluster"] = labels
# Đánh giá chất lượng gom cụm: Silhouette càng gần 1 càng tốt, WCSS (inertia) càng nhỏ càng chặt
print("WCSS (inertia):", round(model.inertia_, 4))
print("Silhouette    :", round(silhouette_score(X, labels), 4))
# Trực quan hóa bằng giá trị gốc (chưa chuẩn hóa) để dễ đọc trục
visualizeKMeans(df2.loc[:, columns].values, y_kmeans,
                cluster,
                "Clusters of Customers - GenreCount X AvgMinutes",
                "GenreCount (số thể loại đã mua)",
                "AvgMinutes (thời lượng trung bình track - phút)",
                colors)
showClusterSummary(df2, columns)
showCustomerDetailByCluster(conn, df2, cluster)
# Xuất Excel: mỗi cụm 1 sheet + sheet Thống kê
dfDetail = getClusterDetail(conn, df2, columns)
exportClustersToExcel(dfDetail, columns, os.path.join(EXCEL_DIR, "CustomerCluster1_Track.xlsx"))
# Lưu mô hình và thông tin chuẩn hóa để dùng lại trong using_model.py, export_results.py, giao diện
modelFile, scalerFile = getModelPaths("kmeans_track")
FileUtil.savemodel(model, modelFile)
FileUtil.savemodel({"columns": columns, "min": minValues, "max": maxValues}, scalerFile)
print("Đã lưu mô hình:", modelFile)
closeConnection(conn)
