import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PyQt6 import QtWidgets
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QMessageBox
from sklearn.metrics import silhouette_score

from Chinook_Customer_Clustering.Utils.ClusterUtil import BASE_DIR, getConnect, closeConnection, queryDataset, \
    getCustomerFeatures, getModelPaths, minMaxScale, runKMeans
from Chinook_Customer_Clustering.Utils.FileUtil import FileUtil
from Chinook_Customer_Clustering.Ui.ChinookMainWindow import Ui_MainWindow


class ChinookMainWindowEx(Ui_MainWindow):
    def initUi(self):
        self.conn = None
        self.df = None          # bảng thuộc tính của 59 customer
        self.model = None       # mô hình KMeans đang dùng
        self.scaler = None      # {"columns", "min", "max"} dùng để chuẩn hóa
        self.colors = ["red", "green", "blue", "purple", "black", "pink", "orange", "brown", "gray", "olive"]
        # 3 lựa chọn trong comboBoxModel: các cột gom cụm + tên file lưu mô hình
        self.modelInfos = [
            {"columns": ["GenreCount", "AvgMinutes"], "file": "kmeans_track"},
            {"columns": ["AlbumCount", "AlbumPopularity"], "file": "kmeans_album"},
            {"columns": ["GenreCount", "AvgMinutes", "AlbumCount"], "file": "kmeans_track_album"},
        ]

    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.MainWindow = MainWindow
        self.initUi()
        self.setupSignalAndSlot()
        self.setupPlot()

    def showWindow(self):
        self.MainWindow.show()

    def setupSignalAndSlot(self):
        self.pushButtonLoadData.clicked.connect(self.loadData)
        self.pushButtonElbow.clicked.connect(self.showElbow)
        self.pushButtonTrain.clicked.connect(self.trainModel)
        self.pushButtonSaveModel.clicked.connect(self.saveModel)
        self.pushButtonLoadModel.clicked.connect(self.loadModel)
        self.pushButtonSearch.clicked.connect(self.searchCustomer)
        self.pushButtonPredict.clicked.connect(self.predictNewCustomer)
        self.comboBoxModel.currentIndexChanged.connect(self.changeModel)

    def setupPlot(self):
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self.MainWindow)

        self.verticalLayoutPlot.addWidget(self.toolbar)
        self.verticalLayoutPlot.addWidget(self.canvas)

    def showMessage(self, text):
        self.msgBox = QMessageBox()
        self.msgBox.setText(text)
        self.msgBox.exec()

    def showDataIntoTableWidget(self, table, df):
        table.setRowCount(0)
        table.setColumnCount(len(df.columns))
        for i in range(len(df.columns)):
            columnHeader = df.columns[i]
            table.setHorizontalHeaderItem(i, QtWidgets.QTableWidgetItem(str(columnHeader)))
        row = 0
        # astype(object) để số nguyên không bị đổi thành số thực (1 -> 1.0)
        for item in df.astype(object).iloc:
            arr = item.values.tolist()
            table.insertRow(row)
            j = 0
            for data in arr:
                table.setItem(row, j, QtWidgets.QTableWidgetItem(str(data)))
                j = j + 1
            row = row + 1
        table.resizeColumnsToContents()

    def getColumns(self):
        return self.modelInfos[self.comboBoxModel.currentIndex()]["columns"]

    # ------------------------------------------------------------
    # 1. Load Data: đọc Chinook, tạo thuộc tính cho 59 customer
    # ------------------------------------------------------------
    def loadData(self):
        # Đường dẫn tương đối (vd Dataset/Chinook_Sqlite.sqlite) được tính từ thư mục gốc project
        databasePath = self.lineEditDatabase.text().strip()
        if not os.path.isabs(databasePath):
            databasePath = os.path.join(BASE_DIR, databasePath)
        if not os.path.exists(databasePath):
            self.showMessage("Không tìm thấy file database: " + databasePath)
            return
        closeConnection(self.conn)
        self.conn = getConnect(databasePath)
        dfFeature = getCustomerFeatures(self.conn)
        sql = "select CustomerId, FirstName || ' ' || LastName as FullName, Country from Customer"
        dfCustomer = queryDataset(self.conn, sql)
        self.df = dfCustomer.merge(dfFeature, on="CustomerId")

        sql = "select Invoice.CustomerId, Album.AlbumId, Album.Title, Artist.Name as Artist " \
              "from Invoice, InvoiceLine, Track, Album, Artist " \
              "where Invoice.InvoiceId = InvoiceLine.InvoiceId and InvoiceLine.TrackId = Track.TrackId " \
              "and Track.AlbumId = Album.AlbumId and Album.ArtistId = Artist.ArtistId"
        self.dfAlbum = queryDataset(self.conn, sql)

        self.model = None
        self.showDataIntoTableWidget(self.tableWidgetCustomer, self.df)
        self.labelStatus.setText(f"Đã tải dữ liệu: {len(self.df)} customer")

    def checkData(self):
        if self.df is None:
            self.showMessage("Hãy bấm Load Data trước")
            return False
        return True

    def checkModel(self):
        if self.model is None:
            self.showMessage("Hãy bấm Train Model hoặc Load Model trước")
            return False
        return True

    def changeModel(self):
        # đổi lựa chọn mô hình thì phải train/load lại
        self.model = None
        self.scaler = None
        if self.df is not None:
            self.df = self.df.drop(columns=["Cluster"], errors="ignore")
            self.showDataIntoTableWidget(self.tableWidgetCustomer, self.df)
        self.labelStatus.setText("Đã đổi mô hình, hãy Train Model hoặc Load Model")

    # ------------------------------------------------------------
    # 2. Elbow: vẽ đồ thị chọn K vào verticalLayoutPlot
    # ------------------------------------------------------------
    def showElbow(self):
        if not self.checkData():
            return
        columns = self.getColumns()
        dfScaled, minValues, maxValues = minMaxScale(self.df, columns)
        X = dfScaled.loc[:, columns].values
        inertia = []
        for n in range(1, 11):
            y_kmeans, centroids, labels, model = runKMeans(X, n)
            inertia.append(model.inertia_)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(np.arange(1, 11), inertia, 'o')
        ax.plot(np.arange(1, 11), inertia, '-', alpha=0.5)
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Cluster sum of squared distances')
        ax.set_title("Elbow Method - " + " X ".join(columns))
        ax.grid()
        self.canvas.draw()

    # ------------------------------------------------------------
    # 3. Train Model: chạy K-Means với K trên spinBoxK
    # ------------------------------------------------------------
    def trainModel(self):
        if not self.checkData():
            return
        columns = self.getColumns()
        cluster = self.spinBoxK.value()
        dfScaled, minValues, maxValues = minMaxScale(self.df, columns)
        X = dfScaled.loc[:, columns].values

        y_kmeans, centroids, labels, model = runKMeans(X, cluster)
        self.model = model
        self.scaler = {"columns": columns, "min": minValues, "max": maxValues}
        self.showResult()
        self.labelStatus.setText(f"Train xong: K = {cluster}  |  Silhouette = {self.silhouette:.3f}")

    # Chuẩn hóa bằng min, max đã lưu rồi predict cụm cho 59 customer
    def predictAll(self):
        columns = self.scaler["columns"]
        dfScaled = (self.df[columns] - self.scaler["min"]) / (self.scaler["max"] - self.scaler["min"])
        X = dfScaled.values
        labels = self.model.predict(X)
        self.silhouette = silhouette_score(X, labels)
        self.df["Cluster"] = labels + 1          # hiển thị cụm bắt đầu từ 1 giống thầy

    def showResult(self, highlightCustomerId=None):
        self.predictAll()
        self.showDataIntoTableWidget(self.tableWidgetCustomer, self.df)
        self.showSummary()
        self.visualizeKMeans(highlightCustomerId)

    def showSummary(self):
        columns = self.scaler["columns"]
        dfSummary = self.df.groupby("Cluster")[columns].mean().round(2)
        dfSummary["NumberOfCustomers"] = self.df.groupby("Cluster")["CustomerId"].count()
        dfSummary = dfSummary.reset_index()
        self.showDataIntoTableWidget(self.tableWidgetSummary, dfSummary)

    def visualizeKMeans(self, highlightCustomerId=None):
        columns = self.scaler["columns"]
        cluster = self.model.n_clusters
        self.figure.clear()
        if len(columns) == 2:
            ax = self.figure.add_subplot(111)
        else:
            ax = self.figure.add_subplot(111, projection="3d")
        for i in range(cluster):
            dfCluster = self.df[self.df["Cluster"] == i + 1]
            values = [dfCluster[column] for column in columns]
            ax.scatter(*values, s=60, c=self.colors[i], label='Cluster %i' % (i + 1))
        if highlightCustomerId is not None:
            dfOne = self.df[self.df["CustomerId"] == highlightCustomerId]
            values = [dfOne[column] for column in columns]
            ax.scatter(*values, s=300, facecolors="none", edgecolors="orange", linewidths=3,
                       label="Customer %i" % highlightCustomerId)
        ax.set_title("Clusters of Customers - " + " X ".join(columns))
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        if len(columns) == 3:
            ax.set_zlabel(columns[2])
        ax.legend()
        self.canvas.draw()

    # ------------------------------------------------------------
    # 4. Save Model / Load Model (dùng FileUtil giống bài Housing)
    # ------------------------------------------------------------
    def getModelFileNames(self):
        name = self.modelInfos[self.comboBoxModel.currentIndex()]["file"]
        return getModelPaths(name)

    def saveModel(self):
        if not self.checkModel():
            return
        modelFile, scalerFile = self.getModelFileNames()
        FileUtil.savemodel(self.model, modelFile)
        FileUtil.savemodel(self.scaler, scalerFile)
        self.showMessage("Đã lưu mô hình: Trained/" + os.path.basename(modelFile))

    def loadModel(self):
        if not self.checkData():
            return
        modelFile, scalerFile = self.getModelFileNames()
        if not os.path.exists(modelFile) or not os.path.exists(scalerFile):
            self.showMessage("Chưa có file Trained/" + os.path.basename(modelFile) + ", hãy Train Model rồi Save Model trước")
            return
        self.model = FileUtil.loadmodel(modelFile)
        self.scaler = FileUtil.loadmodel(scalerFile)
        self.spinBoxK.setValue(self.model.n_clusters)
        self.showResult()
        self.labelStatus.setText(f"Đã tải mô hình: Trained/{os.path.basename(modelFile)}  |  K = {self.model.n_clusters}"
                                 f"  |  Silhouette = {self.silhouette:.3f}")

    # ------------------------------------------------------------
    # 5. Tra cứu Customer: xem cụm + gợi ý album
    #    Gợi ý = album được nhiều customer CÙNG CỤM mua nhất mà customer này chưa mua
    # ------------------------------------------------------------
    def searchCustomer(self):
        if not self.checkData() or not self.checkModel():
            return
        customerId = self.spinBoxCustomerId.value()
        dfOne = self.df[self.df["CustomerId"] == customerId]
        cluster = dfOne["Cluster"].values[0]
        self.labelSearchResult.setText(f"{dfOne['FullName'].values[0]} ({dfOne['Country'].values[0]})"
                                       f"  →  Cluster {cluster}")

        sameClusterIds = self.df[self.df["Cluster"] == cluster]["CustomerId"]
        boughtAlbumIds = self.dfAlbum[self.dfAlbum["CustomerId"] == customerId]["AlbumId"].unique()
        dfCandidate = self.dfAlbum[self.dfAlbum["CustomerId"].isin(sameClusterIds) &
                                   ~self.dfAlbum["AlbumId"].isin(boughtAlbumIds)]
        dfRecommend = dfCandidate.groupby(["Title", "Artist"])["CustomerId"].nunique() \
            .reset_index(name="CustomersInCluster") \
            .sort_values(["CustomersInCluster", "Title"], ascending=[False, True]).head(5)
        self.showDataIntoTableWidget(self.tableWidgetRecommend, dfRecommend)

        # tô vàng dòng customer trong bảng và khoanh tròn trên biểu đồ
        self.showDataIntoTableWidget(self.tableWidgetCustomer, self.df)
        row = dfOne.index[0]
        for j in range(self.tableWidgetCustomer.columnCount()):
            self.tableWidgetCustomer.item(row, j).setBackground(QColor("yellow"))
        self.tableWidgetCustomer.scrollToItem(self.tableWidgetCustomer.item(row, 0))
        self.visualizeKMeans(customerId)

    # ------------------------------------------------------------
    # 6. Dự đoán cụm cho Customer mới (mô hình quan tâm TRACK: GenreCount, AvgMinutes)
    # ------------------------------------------------------------
    def predictNewCustomer(self):
        try:
            genreCount = float(self.lineEditGenreCount.text())
            avgMinutes = float(self.lineEditAvgMinutes.text())
        except ValueError:
            self.showMessage("GenreCount và AvgMinutes phải là số")
            return
        if self.model is not None and self.scaler["columns"] == ["GenreCount", "AvgMinutes"]:
            model, scaler = self.model, self.scaler
        else:
            # mô hình đang chọn không phải TRACK -> tải mô hình TRACK đã lưu
            trackModelFile, trackScalerFile = getModelPaths("kmeans_track")
            if not os.path.exists(trackModelFile):
                self.showMessage("Hãy chọn mô hình 'Quan tâm TRACK', Train Model rồi Save Model trước")
                return
            model = FileUtil.loadmodel(trackModelFile)
            scaler = FileUtil.loadmodel(trackScalerFile)
        dfNew = pd.DataFrame([[genreCount, avgMinutes]], columns=["GenreCount", "AvgMinutes"])
        dfScaled = (dfNew - scaler["min"]) / (scaler["max"] - scaler["min"])
        cluster = model.predict(dfScaled.values)[0] + 1
        self.lineEditPredictResult.setText(f"Cluster {cluster}")
