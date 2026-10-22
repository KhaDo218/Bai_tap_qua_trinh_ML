# THỰC NGHIỆM 3: Gom cụm Customer theo cả TRACK và ALBUM (3 thuộc tính) + trực quan hóa 3D
# Thuộc tính: GenreCount, AvgMinutes, AlbumCount
import os

from sklearn.metrics import silhouette_score

from Chinook_Customer_Clustering.Utils.ClusterUtil import *
from Chinook_Customer_Clustering.Utils.ExportUtil import exportClustersToExcel
from Chinook_Customer_Clustering.Utils.FileUtil import FileUtil

conn = getConnect(DATABASE_PATH)

df2 = getCustomerFeatures(conn)
print(df2.head())

columns = ["GenreCount", "AvgMinutes", "AlbumCount"]

dfScaled, minValues, maxValues = minMaxScale(df2, columns)

cluster = elbowMethod(dfScaled, columns)
print("Số cụm được chọn: K =", cluster)

X = dfScaled.loc[:, columns].values

y_kmeans, centroids, labels, model = runKMeans(X, cluster)
print(y_kmeans)
print(centroids)
print(labels)
df2["cluster"] = labels

# Đánh giá chất lượng gom cụm: Silhouette càng gần 1 càng tốt, WCSS (inertia) càng nhỏ càng chặt
print("WCSS (inertia):", round(model.inertia_, 4))
print("Silhouette    :", round(silhouette_score(X, labels), 4))

hover_data = df2.columns
visualize3DKmeans(df2, columns, hover_data, cluster)

showClusterSummary(df2, columns)
showCustomerDetailByCluster(conn, df2, cluster)

dfDetail = getClusterDetail(conn, df2, columns)
exportClustersToExcel(dfDetail, columns, os.path.join(EXCEL_DIR, "CustomerCluster3_TrackAlbum.xlsx"))

modelFile, scalerFile = getModelPaths("kmeans_track_album")
FileUtil.savemodel(model, modelFile)
FileUtil.savemodel({"columns": columns, "min": minValues, "max": maxValues}, scalerFile)
print("Đã lưu mô hình:", modelFile)

closeConnection(conn)
