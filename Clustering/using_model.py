# SỬ DỤNG MÔ HÌNH ĐÃ LƯU (chạy CustomerCluster1.py và CustomerCluster2.py trước để có file trong Trained)
import pandas as pd

from Chinook_Customer_Clustering.Utils.ClusterUtil import *
from Chinook_Customer_Clustering.Utils.FileUtil import FileUtil

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# 1. Tải mô hình và thông tin chuẩn hóa
trackModelFile, trackScalerFile = getModelPaths("kmeans_track")
albumModelFile, albumScalerFile = getModelPaths("kmeans_album")
trackModel = FileUtil.loadmodel(trackModelFile)
trackScaler = FileUtil.loadmodel(trackScalerFile)
albumModel = FileUtil.loadmodel(albumModelFile)
albumScaler = FileUtil.loadmodel(albumScalerFile)


# 2. Hàm chuẩn hóa dữ liệu mới bằng đúng min, max lúc train rồi dự đoán cụm
def predictCluster(model, scaler, df):
    columns = scaler["columns"]
    dfScaled = (df[columns] - scaler["min"]) / (scaler["max"] - scaler["min"])
    return model.predict(dfScaled.values)


conn = getConnect(DATABASE_PATH)
df = getCustomerFeatures(conn)
df["TrackCluster"] = predictCluster(trackModel, trackScaler, df) + 1   # cụm đánh số từ 1
df["AlbumCluster"] = predictCluster(albumModel, albumScaler, df) + 1
print(df)

# 3. Xem 1 customer thuộc cụm nào
customerId = int(input("Nhập CustomerId (1 - 59): "))
dfOne = df[df["CustomerId"] == customerId]
print(dfOne.to_string(index=False))
trackCluster = dfOne["TrackCluster"].values[0]
albumCluster = dfOne["AlbumCluster"].values[0]
print("Cụm theo quan tâm TRACK: Cluster", trackCluster)
print("Cụm theo quan tâm ALBUM: Cluster", albumCluster)

# 4. Gợi ý album: album được nhiều customer cùng cụm ALBUM mua nhất mà customer này chưa mua
dfAlbum = getAlbumPurchaseDataset(conn)
sameClusterIds = df[df["AlbumCluster"] == albumCluster]["CustomerId"]
dfRecommend = recommendAlbums(dfAlbum, sameClusterIds, customerId, 5)
print("Top 5 album gợi ý cho customer", customerId)
print(dfRecommend.to_string(index=False))

# 5. Dự đoán cụm cho 1 customer MỚI (tự nhập các thuộc tính)
genreCount = float(input("Nhập GenreCount của customer mới (vd 5): "))
avgMinutes = float(input("Nhập AvgMinutes của customer mới (vd 4.5): "))
dfNew = pd.DataFrame([[genreCount, avgMinutes]], columns=["GenreCount", "AvgMinutes"])
newCluster = predictCluster(trackModel, trackScaler, dfNew)
print("Customer mới thuộc cụm TRACK: Cluster", newCluster[0] + 1)

closeConnection(conn)
