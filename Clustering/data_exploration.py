import matplotlib.pyplot as plt

from Chinook_Customer_Clustering.Utils.ClusterUtil import DATABASE_PATH, getConnect, queryDataset, closeConnection

conn = getConnect(DATABASE_PATH)

# 1. Các bảng có trong cơ sở dữ liệu Chinook và số dòng
sql1 = "select name from sqlite_master where type='table'"
df1 = queryDataset(conn, sql1)
for tableName in df1["name"]:
    df = queryDataset(conn, f"select count(*) as Total from [{tableName}]")
    print(tableName, ":", df["Total"][0], "dòng")

# 2. Doanh thu theo tháng -> hầu như không đổi (dữ liệu giả lập)
#    => bài toán dự báo doanh thu không có nhiều ý nghĩa
sql2 = "select strftime('%Y-%m', InvoiceDate) as Month, round(sum(Total), 2) as Revenue " \
       "from Invoice group by Month"
df2 = queryDataset(conn, sql2)
print(df2.head())
print(df2["Revenue"].describe())
print("Số tháng có doanh thu đúng 37.62:", (df2["Revenue"] == 37.62).sum(), "/", len(df2))

plt.figure(figsize=(15, 6))
plt.plot(df2["Month"], df2["Revenue"], 'o-')
plt.xticks(rotation=90)
plt.title("Doanh thu theo tháng")
plt.show()

# 3. Hành vi mua của từng customer
#    => ai cũng mua khoảng 38 track trong 7 hóa đơn, nhưng số thể loại / số album lại khác nhau
#    => phù hợp để GOM CỤM khách hàng theo mức độ quan tâm tới Track và Album
sql3 = "select Invoice.CustomerId, count(distinct Invoice.InvoiceId) as InvoiceCount, " \
       "count(InvoiceLine.TrackId) as TrackCount, " \
       "count(distinct Track.GenreId) as GenreCount, " \
       "count(distinct Track.AlbumId) as AlbumCount " \
       "from Invoice, InvoiceLine, Track " \
       "where Invoice.InvoiceId = InvoiceLine.InvoiceId and InvoiceLine.TrackId = Track.TrackId " \
       "group by Invoice.CustomerId"
df3 = queryDataset(conn, sql3)
print(df3.head())
print(df3.describe())

# 4. Mỗi track được mua bao nhiêu lần
#    => track hầu như chỉ được mua 1 lần nên rất khó làm bài toán gợi ý theo từng track
sql4 = "select TrackId, count(*) as TimesSold from InvoiceLine group by TrackId"
df4 = queryDataset(conn, sql4)
print(df4["TimesSold"].value_counts())

# 5. Top 10 thể loại được mua nhiều nhất
sql5 = "select Genre.Name as Genre, count(*) as TrackSold " \
       "from InvoiceLine, Track, Genre " \
       "where InvoiceLine.TrackId = Track.TrackId and Track.GenreId = Genre.GenreId " \
       "group by Genre.Name order by TrackSold desc limit 10"
df5 = queryDataset(conn, sql5)
print(df5)

plt.figure(figsize=(10, 6))
plt.barh(df5["Genre"], df5["TrackSold"])
plt.gca().invert_yaxis()
plt.title("Top 10 thể loại được mua nhiều nhất")
plt.show()

closeConnection(conn)
