import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.cluster import KMeans

# 0. Đường dẫn dùng chung, tính từ vị trí file này nên chạy file ở thư mục nào cũng đúng
#    (file này nằm trong Utils/, thư mục gốc của project là thư mục cha)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_PATH = os.path.join(BASE_DIR, "Dataset", "Chinook_Sqlite.sqlite")
TRAINED_DIR = os.path.join(BASE_DIR, "Trained")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
EXCEL_DIR = os.path.join(RESULTS_DIR, "excel")
HTML_DIR = os.path.join(RESULTS_DIR, "html")
IMAGES_DIR = os.path.join(RESULTS_DIR, "images")
# 3 mô hình (3 thực nghiệm): cột gom cụm + tên file lưu trong Trained
EXPERIMENTS = [
    {"name": "Thực nghiệm 1 - Quan tâm TRACK", "columns": ["GenreCount", "AvgMinutes"],
     "file": "kmeans_track"},
    {"name": "Thực nghiệm 2 - Quan tâm ALBUM", "columns": ["AlbumCount", "AlbumPopularity"],
     "file": "kmeans_album"},
    {"name": "Thực nghiệm 3 - TRACK + ALBUM", "columns": ["GenreCount", "AvgMinutes", "AlbumCount"],
     "file": "kmeans_track_album"},
]
def getModelPaths(fileName):
    return (os.path.join(TRAINED_DIR, fileName + "_model.zip"),
            os.path.join(TRAINED_DIR, fileName + "_scaler.zip"))
# 1. Kết nối và đóng kết nối cơ sở dữ liệu SQLite
def getConnect(databasePath=DATABASE_PATH):
    try:
        conn = sqlite3.connect(databasePath)
        return conn
    except sqlite3.Error as e:
        print("Error = ", e)
    return None
def closeConnection(conn):
    if conn != None:
        conn.close()

# 2. Truy vấn và trả về DataFrame (có tên cột)
def queryDataset(conn, sql):
    cursor = conn.cursor()
    cursor.execute(sql)
    df = pd.DataFrame(cursor.fetchall())
    df.columns = [column[0] for column in cursor.description]
    cursor.close()
    return df


# 3. Lấy dữ liệu mua hàng: mỗi dòng là 1 track mà 1 customer đã mua
def getPurchaseDataset(conn):
    sql = "select Invoice.CustomerId, InvoiceLine.TrackId, InvoiceLine.UnitPrice, " \
          "Track.Milliseconds, Track.AlbumId, Track.GenreId " \
          "from Invoice, InvoiceLine, Track " \
          "where Invoice.InvoiceId = InvoiceLine.InvoiceId " \
          "and InvoiceLine.TrackId = Track.TrackId"
    df = queryDataset(conn, sql)
    return df


# Thông tin mô tả customer (không đưa vào mô hình): tên, quốc gia, tổng chi tiêu
def getCustomerInfo(conn):
    sql = "select Customer.CustomerId, Customer.FirstName || ' ' || Customer.LastName as FullName, " \
          "Customer.Country, Customer.City, Customer.Email, round(sum(Invoice.Total), 2) as TotalSpent " \
          "from Customer, Invoice " \
          "where Customer.CustomerId = Invoice.CustomerId " \
          "group by Customer.CustomerId"
    df = queryDataset(conn, sql)
    return df


# Album + nghệ sĩ + thể loại của từng lượt mua (dùng cho mô tả cụm và gợi ý album)
def getAlbumPurchaseDataset(conn):
    sql = "select Invoice.CustomerId, Album.AlbumId, Album.Title, Artist.Name as Artist, Genre.Name as Genre " \
          "from Invoice, InvoiceLine, Track, Album, Artist, Genre " \
          "where Invoice.InvoiceId = InvoiceLine.InvoiceId and InvoiceLine.TrackId = Track.TrackId " \
          "and Track.AlbumId = Album.AlbumId and Album.ArtistId = Artist.ArtistId " \
          "and Track.GenreId = Genre.GenreId"
    df = queryDataset(conn, sql)
    return df


# 4. Tạo các thuộc tính (cột) thể hiện mức độ quan tâm tới Track và Album của từng Customer
#    - GenreCount     : số thể loại khác nhau của các track đã mua   (quan tâm TRACK - đa dạng hay tập trung)
#    - AvgMinutes     : thời lượng trung bình (phút) các track đã mua (quan tâm TRACK - bài hát ngắn hay video dài)
#    - AlbumCount     : số album khác nhau đã mua track              (quan tâm ALBUM - mua rải rác nhiều album hay gom vào ít album)
#    - AlbumPopularity: độ phổ biến trung bình của các album đã mua  (quan tâm ALBUM - album "hot" hay album ít người mua)
#                       độ phổ biến của 1 album = số customer đã mua ít nhất 1 track của album đó
def getCustomerFeatures(conn):
    dfPurchase = getPurchaseDataset(conn)

    # độ phổ biến của từng album
    albumPopularity = dfPurchase.groupby("AlbumId")["CustomerId"].nunique()

    rows = []
    for customerId, dfCustomer in dfPurchase.groupby("CustomerId"):
        albums = dfCustomer["AlbumId"].unique()
        rows.append([customerId,
                     dfCustomer["GenreId"].nunique(),
                     round(dfCustomer["Milliseconds"].mean() / 60000, 2),
                     len(albums),
                     round(albumPopularity[albums].mean(), 2)])

    df = pd.DataFrame(rows, columns=["CustomerId", "GenreCount", "AvgMinutes",
                                     "AlbumCount", "AlbumPopularity"])
    return df


# 5. Chuẩn hóa Min-Max về đoạn [0, 1] để các cột có cùng thang đo trước khi tính khoảng cách Euclid
#    Công thức: x' = (x - min) / (max - min)
def minMaxScale(df, columns):
    minValues = df[columns].min()
    maxValues = df[columns].max()
    dfScaled = (df[columns] - minValues) / (maxValues - minValues)
    return dfScaled, minValues, maxValues


def showHistogram(df, columns):
    plt.figure(figsize=(7, 8))
    n = 0
    for column in columns:
        n += 1
        plt.subplot(len(columns), 1, n)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sns.histplot(df[column], bins=32, kde=True)
        plt.title(f'Histogram of {column}')
    plt.show()


# 6. Elbow + thuật toán Kneedle tự tìm số cụm
#    - Chuẩn hóa K và inertia về [0, 1]; đường thẳng nối điểm đầu - điểm cuối là y = 1 - x
#    - Độ "gập" của mỗi K: d = (1 - x) - y ; K có d lớn nhất là điểm khuỷu tay (knee)
#    - Khoảng elbow: các K có d >= tolerance * d lớn nhất (mặc định 90%)
#    Bài làm chọn K = điểm gập lớn nhất (không ưu tiên K lớn hơn trong khoảng elbow),
#    vì với dữ liệu Chinook, K lớn hơn trong khoảng elbow làm Silhouette giảm
#    (thực nghiệm 1: K = 3 -> 0.434, K = 4 -> 0.399) và tách nhỏ cụm mất ý nghĩa.
def findElbow(X, maxK=10, tolerance=0.9):
    kRange = np.arange(1, maxK + 1)
    inertia = []
    for n in kRange:
        model = KMeans(n_clusters=n,
                       init='k-means++',
                       max_iter=500,
                       n_init=10,
                       random_state=42)
        model.fit(X)
        inertia.append(model.inertia_)
    inertia = np.array(inertia)
    xNorm = (kRange - kRange.min()) / (kRange.max() - kRange.min())
    yNorm = (inertia - inertia.min()) / (inertia.max() - inertia.min() + 1e-12)
    distance = (1 - xNorm) - yNorm
    optimalK = int(kRange[distance.argmax()])
    elbowZone = kRange[distance >= tolerance * distance.max()]
    return kRange, inertia, elbowZone, optimalK

def elbowMethod(df, columnsForElbow, maxK=10, tolerance=0.9):
    X = df.loc[:, columnsForElbow].values
    kRange, inertia, elbowZone, optimalK = findElbow(X, maxK, tolerance)

    print("Inertia theo K:", dict(zip(kRange.tolist(), np.round(inertia, 2).tolist())))
    print("Khoảng elbow:", elbowZone.tolist(), "=> điểm gập lớn nhất K =", optimalK)

    plt.figure(figsize=(15, 6))
    plt.plot(kRange, inertia, 'o')
    plt.plot(kRange, inertia, '-', alpha=0.5)
    if len(elbowZone) > 1:
        plt.axvspan(elbowZone.min() - 0.3, elbowZone.max() + 0.3, color='orange', alpha=0.15,
                    label=f'Khoảng elbow: K = {elbowZone.min()} - {elbowZone.max()}')
    plt.axvline(x=optimalK, color='red', linestyle='--', linewidth=1.5,
                label=f'Điểm gập lớn nhất K = {optimalK}')
    plt.plot(optimalK, inertia[optimalK - 1], 'o', color='red', markersize=12)
    plt.xticks(kRange)
    plt.xlabel('Number of Clusters'), plt.ylabel('Cluster sum of squared distances')
    plt.title('Elbow Method (' + ' X '.join(columnsForElbow) + ')')
    plt.legend()
    plt.show()
    return optimalK


def runKMeans(X, cluster):
    model = KMeans(n_clusters=cluster,
                   init='k-means++',
                   max_iter=500,
                   n_init=10,
                   random_state=42)
    model.fit(X)
    labels = model.labels_
    centroids = model.cluster_centers_
    y_kmeans = model.predict(X)
    return y_kmeans, centroids, labels, model


def visualizeKMeans(X, y_kmeans, cluster, title, xlabel, ylabel, colors):
    plt.figure(figsize=(10, 10))
    for i in range(cluster):
        plt.scatter(X[y_kmeans == i, 0],
                    X[y_kmeans == i, 1],
                    s=100,
                    c=colors[i],
                    label='Cluster %i' % (i + 1))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def visualize3DKmeans(df, columns, hover_data, cluster):
    df = df.copy()
    df["cluster"] = df["cluster"] + 1      # đánh số cụm từ 1 giống biểu đồ 2D
    fig = px.scatter_3d(df,
                        x=columns[0],
                        y=columns[1],
                        z=columns[2],
                        color='cluster',
                        hover_data=hover_data,
                        category_orders={"cluster": range(1, cluster + 1)},
                        )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()


# 7. Đặt tên cụm tự động: so sánh trung bình của cụm với trung bình toàn bộ khách theo z-score,
#    lấy thuộc tính lệch nhiều nhất (và thuộc tính thứ hai nếu lệch >= 0.5 độ lệch chuẩn)
FEATURE_WORDS = {
    "GenreCount": ("Nghe đa dạng thể loại", "Tập trung ít thể loại"),
    "AvgMinutes": ("Mua nội dung video dài", "Nghe track ngắn"),
    "AlbumCount": ("Mua rải rác nhiều album", "Gom vào ít album"),
    "AlbumPopularity": ("Theo album phổ biến", "Săn album ít người biết"),
}


def nameClusters(df, columns, labelColumn="cluster"):
    mean = df[columns].mean()
    std = df[columns].std()
    names = {}
    for c in sorted(df[labelColumn].unique()):
        z = (df[df[labelColumn] == c][columns].mean() - mean) / std
        order = z.abs().sort_values(ascending=False).index
        words = []
        for col in order[:2]:
            if len(words) == 0 or abs(z[col]) >= 0.5:
                words.append(FEATURE_WORDS[col][0] if z[col] > 0 else FEATURE_WORDS[col][1])
        names[c] = " · ".join(words)
    return names


# 8. Thống kê trung bình các thuộc tính theo từng cụm (để đặt tên / mô tả cụm)
#    Cụm được đánh số từ 1 cho khớp với chú thích 'Cluster 1, 2, ...' trên biểu đồ
def showClusterSummary(df, columns):
    dfShow = df.copy()
    dfShow["cluster"] = dfShow["cluster"] + 1
    summary = dfShow.groupby("cluster")[columns].mean().round(2)
    summary["NumberOfCustomers"] = dfShow.groupby("cluster")["CustomerId"].count()
    print(summary)
    return summary

# 9. Bài tập 1 (slide 07): in danh sách chi tiết Customer của từng cụm ra màn hình console
def showCustomerDetailByCluster(conn, df, cluster):
    sql = "select CustomerId, FirstName, LastName, Company, City, Country, Email from Customer"
    dfCustomer = queryDataset(conn, sql)
    dfDetail = dfCustomer.merge(df[["CustomerId", "cluster"]], on="CustomerId")
    dfDetail["cluster"] = dfDetail["cluster"] + 1
    for i in range(cluster):
        dfCluster = dfDetail[dfDetail["cluster"] == i + 1]
        print("-" * 100)
        print(f"Cluster {i + 1}: có {len(dfCluster)} customer")
        print(dfCluster.to_string(index=False))

# 10. Gợi ý album: album được nhiều customer cùng cụm mua nhất mà customer này chưa mua
def recommendAlbums(dfAlbum, sameClusterIds, customerId, topN=5):
    boughtAlbumIds = dfAlbum[dfAlbum["CustomerId"] == customerId]["AlbumId"].unique()
    dfCandidate = dfAlbum[dfAlbum["CustomerId"].isin(sameClusterIds) &
                          ~dfAlbum["AlbumId"].isin(boughtAlbumIds)]
    dfRecommend = dfCandidate.groupby(["AlbumId", "Title", "Artist"])["CustomerId"].nunique() \
        .reset_index(name="CustomersInCluster") \
        .sort_values(["CustomersInCluster", "Title"], ascending=[False, True])
    return dfRecommend.head(topN)

# 11. Bảng chi tiết để xuất Excel / HTML: thông tin customer + thuộc tính + cụm (đánh số từ 1) + tên cụm
#     df phải có cột "cluster" (nhãn gốc của KMeans, bắt đầu từ 0)
def getClusterDetail(conn, df, columns):
    names = nameClusters(df, columns)
    dfDetail = getCustomerInfo(conn).merge(df, on="CustomerId")
    dfDetail["ClusterName"] = dfDetail["cluster"].map(lambda c: names[c])
    dfDetail["cluster"] = dfDetail["cluster"] + 1
    return dfDetail.sort_values(["cluster", "CustomerId"]).reset_index(drop=True)
