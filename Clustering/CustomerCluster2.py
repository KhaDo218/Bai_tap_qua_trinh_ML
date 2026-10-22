# THỰC NGHIỆM 2: Gom cụm Customer theo mức độ quan tâm tới ALBUM
# Thuộc tính: AlbumCount (số album khác nhau đã mua), AlbumPopularity (độ phổ biến trung bình album đã mua)
import os

from sklearn.metrics import silhouette_score

from Chinook_Customer_Clustering.Utils.ClusterUtil import *
from Chinook_Customer_Clustering.Utils.ExportUtil import exportClustersToExcel
from Chinook_Customer_Clustering.Utils.FileUtil import FileUtil

conn = getConnect(DATABASE_PATH)

df2 = getCustomerFeatures(conn)
print(df2.head())
print(df2.describe())

columns = ["AlbumCount", "AlbumPopularity"]

showHistogram(df2, columns)

dfScaled, minValues, maxValues = minMaxScale(df2, columns)
print(dfScaled.head())

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

visualizeKMeans(df2.loc[:, columns].values,
                y_kmeans,
                cluster,
                "Clusters of Customers - AlbumCount X AlbumPopularity",
                "AlbumCount (số album đã mua)",
                "AlbumPopularity (độ phổ biến trung bình album đã mua)",
                colors)

showClusterSummary(df2, columns)
showCustomerDetailByCluster(conn, df2, cluster)

dfDetail = getClusterDetail(conn, df2, columns)
exportClustersToExcel(dfDetail, columns, os.path.join(EXCEL_DIR, "CustomerCluster2_Album.xlsx"))

modelFile, scalerFile = getModelPaths("kmeans_album")
FileUtil.savemodel(model, modelFile)
FileUtil.savemodel({"columns": columns, "min": minValues, "max": maxValues}, scalerFile)
print("Đã lưu mô hình:", modelFile)

closeConnection(conn)
