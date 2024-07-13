import sys, time, serial, pygame, lifevision
from PyQt5.QtWidgets import QHBoxLayout, QMessageBox, QStatusBar, QDialog, QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QLCDNumber, QFrame, QDesktopWidget
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer, QTime, QDate
from tkinter import Tk, Message
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

Tm_value=0.0
Hr_value=0
Al_value=0.0

class MyApp(QMainWindow, QDialog, QWidget):
    def __init__(self):
        super().__init__()
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.timeout)
        
        self.time = QTime.currentTime()
        self.date = QDate.currentDate()
        
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.dialog=QDialog()
        self.timer.start()
        self.initUI()
        

    def initUI(self):
        lt = QVBoxLayout()
        pic=QPixmap("assets/LogoLong.png")
        fnt=QFont("D2Coding", 30, QFont.Bold)

        imlbl = QLabel(self)
        imlbl.setGeometry(40, 40, 200, 67)
        imlbl.setPixmap(pic)
        imlbl.setScaledContents(True)

        lbl = QLabel("안전장비 착용감지 시스템", self)
        lbl.setFont(QFont("D2Coding", 50, QFont.Bold))
        lbl.setGeometry(130, 200, 820, 80)

        Ylbl = QLabel(self)
        Ylbl.setFont(fnt)
        Ylbl.setText(self.date.toString(Qt.DefaultLocaleShortDate))
        Ylbl.setGeometry(430, 370, 220, 60)

        self.lcd = QLCDNumber(self)
        self.lcd.display('')
        self.lcd.setFrameStyle(QFrame.NoFrame)
        self.lcd.setDigitCount(8)
        self.lcd.setGeometry(330, 440, 400, 120)
        self.lcd.setSegmentStyle(2)

        self.loglbl =  QLabel('', self)
        self.loglbl.setFont(QFont("D2Coding", 40, QFont.Bold))
        self.loglbl.setGeometry(130, 1500, 820, 160)
        
        self.sbtn = QPushButton('Start', self)
        self.sbtn.setGeometry(470, 950, 140, 55)
        self.sbtn.setFont(fnt)
        self.sbtn.setDefault(True)
        self.sbtn.clicked.connect(self.cmd)
        
        qbtn = QPushButton('Quit', self)
        qbtn.setGeometry(470, 1050, 140, 55)
        qbtn.setFont(fnt)
        qbtn.clicked.connect(self.closeMessage)

        lt.addWidget(lbl)
        lt.addWidget(self.lcd)
        lt.addWidget(self.loglbl)
        lt.addWidget(self.sbtn)
        lt.addWidget(qbtn)

        self.setLayout(lt)
        self.setWindowTitle('안전장구류 인식 시스템')
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.FramelessWindowHint)
        self.statusBar.showMessage(self.date.toString(Qt.DefaultLocaleLongDate))
        self.resize(1080, 1920)
        self.center()
        self.show()

    def timeout(self):
        sender = self.sender()
        currentTime = QTime.currentTime().toString("hh:mm:ss")
        if id(sender) == id(self.timer):
            self.lcd.display(currentTime)

    def center(self):
        qr=self.frameGeometry()
        cp=QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeMessage(self):
        message = QMessageBox.question(self, "Question", "Are you sure you want to quit?")
        if message == QMessageBox.Yes:
            sys.exit()

    def cmd(self):
        global Tm_value, Hr_value, Al_value
        self.sbtn.setText('Running')
        self.sbtn.setEnabled(False)
        Belt_con=False
        Shoes_con=False
        Helmet_con=False
        Pid, username=arduino_rfid()
        self.loglbl.setText('User : '+username+'\nID : '+str(Pid))
        ax=PlotWindow()
        ax.exec_()
        if not (Tm_value==0.0 or Hr_value==0 or Al_value==0.0 or Al_value>200.0 or Hr_value>130 or Tm_value>37.5) :
            Belt_con, Helmet_con, Shoes_con=lifevision.main()
            print(Belt_con, Helmet_con, Shoes_con)
            if not Belt_con or not Helmet_con or not Shoes_con :
                Play("Denied.wav")
        else : 
            Play("Denied.wav")
        
        upload_to_firebase(Pid, username, Belt_con, Shoes_con, Helmet_con, Tm_value, Hr_value, Al_value)
        Be=tr(Belt_con)
        Sh=tr(Shoes_con)
        He=tr(Helmet_con)
        Tm=tr(Tm_value)
        Hr=tr(Hr_value)
        Al=tr(Al_value)
        string='사번 : '+str(Pid)+'\n이름 : '+username+'\n안전대착용여부 : '+Be+'\n안전모착용여부 : '+He+'\n안전화착용여부 : '+Sh+'\n알코올 : '+Al+'\n심박수 : '+Hr+'\n체온 : '+Tm
        QMessageBox.information(self, "Result", string)
        self.loglbl.setText('')
        self.sbtn.setText('Start')
        self.sbtn.setEnabled(True)

def tr(alpha):
    beta=''
    if(str(type(alpha)) == "<class 'bool'>"):
        if(alpha): beta='OK'
        else: beta='Not Detect. 근무 불가'
    elif(str(type(alpha)) == "<class 'float'>"):
        if 37.5<alpha<41.0 or alpha>200.0: beta=str(alpha)+'; 근무 불가'
        elif alpha<1 : beta='검사 X'
        else : beta=str(alpha)
    elif(str(type(alpha)) == "<class 'int'>"):
        if 130<alpha : beta=str(alpha)+'; 근무 불가'
        elif alpha<1 : beta='검사 X'
        else : beta=str(alpha)
    return beta

class PlotWindow(QDialog):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(1080, 800)
        self.setWindowTitle("Bio Plot")
        fnt=QFont("D2Coding", 30, QFont.Bold)

        self.AlBtn = QPushButton("음주\n측정")
        self.AlBtn.setFont(fnt)
        self.AlBtn.clicked.connect(self.AlBtnClicked)
        self.HrBtn = QPushButton("심박수\n측정")
        self.HrBtn.setFont(fnt)
        self.HrBtn.clicked.connect(self.HrBtnClicked)
        self.TmBtn = QPushButton("체온\n측정")
        self.TmBtn.setFont(fnt)
        self.TmBtn.clicked.connect(self.TmBtnClicked)

        self.QBtn = QPushButton("Quit")
        self.QBtn.setFont(QFont("D2Coding", 20, QFont.Bold))
        self.QBtn.clicked.connect(self.QBtnClicked)
        self.QBtn.setEnabled(False)

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.canvas)

        rightLayout = QVBoxLayout()
        rightLayout.addStretch(2)
        rightLayout.addWidget(self.AlBtn)
        rightLayout.addStretch(1)
        rightLayout.addWidget(self.HrBtn)
        rightLayout.addStretch(1)
        rightLayout.addWidget(self.TmBtn)
        rightLayout.addStretch(2)
        rightLayout.addWidget(self.QBtn)
        rightLayout.addStretch(2)

        layout = QHBoxLayout()
        layout.addLayout(leftLayout)
        layout.addLayout(rightLayout)
        layout.setStretchFactor(leftLayout, 1)
        layout.setStretchFactor(rightLayout, 0)
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.FramelessWindowHint)
        self.setLayout(layout)
        self.center()
        self.show()
        
    def AlBtnClicked(self):
        global Al_value
        plt1 = self.fig.add_subplot(5, 1, 1)
        plt1.set_xlim(0, 1000)
        plt1.set_ylim(0, 5)
        plt1.set_xlabel("Blood Alcohol Level")
        plt1.set_yticks([])

        Al_value = Ardu_Al()
        plt1.plot([Al_value], [2.5], 'v', markersize=10, color='dimgrey')

        gradient_image(plt1, direction=1, extent=(0, 1, 0, 1), transform=plt1.transAxes, cmap=plt.cm.BuPu, cmap_range=(0, 0.6), alpha=0.5)

        self.canvas.draw()
        self.AlBtn.setText("알코올\n"+str(Al_value))
        self.AlBtn.setEnabled(False)
        if Al_value>200 : 
            QMessageBox.critical(self, "경고", "음주 후 근무는 위험합니다.")
            self.QBtnClicked()
        if (not self.AlBtn.isEnabled()) and (not self.HrBtn.isEnabled()) and (not self.TmBtn.isEnabled()) : self.QBtn.setDisabled(False)


    def HrBtnClicked(self):
        global Hr_value
        plt2 = self.fig.add_subplot(5, 1, 3)
        plt2.set_xlim(40, 140)
        plt2.set_ylim(0, 5)
        plt2.set_xlabel("Heart Rate")
        plt2.set_yticks([])

        Hr_value = Ardu_Heart()
        plt2.plot([Hr_value], [2.5], 'v', markersize=10, color='dimgrey')

        gradient_image(plt2, direction=1, extent=(0, 1, 0, 1), transform=plt2.transAxes, cmap=plt.cm.hot_r, cmap_range=(0, 0.8), alpha=0.5)

        self.canvas.draw()
        self.HrBtn.setText("심박수\n"+str(Hr_value))
        self.HrBtn.setEnabled(False)

        if Hr_value>130 : 
            QMessageBox.critical(self, "경고", "심박수가 너무 높습니다.")
            self.QBtnClicked()
        if (not self.AlBtn.isEnabled()) and (not self.HrBtn.isEnabled()) and (not self.TmBtn.isEnabled()) : self.QBtn.setDisabled(False)


    def TmBtnClicked(self):
        global Tm_value
        plt3 = self.fig.add_subplot(5, 1, 5)
        plt3.set_xlim(33, 40)
        plt3.set_ylim(0, 5)
        plt3.set_xlabel("Body temperature")
        plt3.set_yticks([])

        Tm_value = Ardu_Tem()
        plt3.plot([Tm_value], [2.5], 'v', markersize=10, color='dimgrey')

        gradient_image(plt3, direction=1, extent=(0, 1, 0, 1), transform=plt3.transAxes, cmap=plt.cm.nipy_spectral, cmap_range=(0.2, 0.9), alpha=0.5)

        self.canvas.draw()
        self.TmBtn.setText("체온\n"+str(Tm_value)+'°C')
        self.TmBtn.setEnabled(False)
        if Tm_value>37.5 : 
            QMessageBox.critical(self, "경고", "체온이 너무 높습니다.")
            self.QBtnClicked()
        if (not self.AlBtn.isEnabled()) and (not self.HrBtn.isEnabled()) and (not self.TmBtn.isEnabled()) : self.QBtn.setEnabled(True)


    def QBtnClicked(self):
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.close)
        self.timer.start()

    def center(self):
        qr=self.frameGeometry()
        cp=QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

def gradient_image(ax, direction=0.3, cmap_range=(0, 1), **kwargs):
    phi = direction * np.pi / 2
    v = np.array([np.cos(phi), np.sin(phi)])
    X = np.array([[v @ [1, 0], v @ [1, 1]],
                  [v @ [0, 0], v @ [0, 1]]])
    a, b = cmap_range
    X = a + (b - a) / X.max() * X
    im = ax.imshow(X, interpolation='bicubic', clim=(0, 1), aspect='auto', **kwargs)
    return im

def Ardu_Al():
    Playtop("Alcohol.wav", 'Breathe into the sensor until you get the result.', de=3500)
    
    time.sleep(1)
    ser=serial.Serial('COM5', 9600)
    if ser.readable():
        for i in range(5):
            val = ser.readline()
            val=val.decode()[:len(val)-1]
            print(val)
    Alch=float(val)
    
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    ser.close()
    top('알코올 수치 : '+str(Alch)+'\n알코올 정상 : 200 이하\n알코올 주의 : 400 이하\n근무 불가 : 500이상\n', 50, 5000)
    return Alch

def Ardu_Tem():
    Playtop("BodyTm.wav", 'Put your wrist on the sensor slightly.', de=3500)
    
    time.sleep(2)
    ser=serial.Serial('COM7', 9600)
    if ser.readable():
        val = ser.readline()
        val=val.decode()[:len(val)-1]
    Temp=float(val)
    
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    ser.close()
    if Temp<30:
        Temp=Ardu_Tem()
    else : top('체온 : '+ val+'°C')
    return Temp

def Ardu_Heart():
    Playtop("HeartBeat.wav", 'Place your finger on the sensor and wait for 10 seconds.', de=3500)

    ser=serial.Serial('COM6', 9600)
    if (ser.readable()):
        for i in range(25):
            val = ser.readline()
            val=val.replace(b'\x00', b'').decode('UTF-8')
            val=val[:len(val)-2]
            if val=='No finger':
                print(val)
            elif val=='Wait' : 
                print(val)
            else : 
                print(val)
                Heart=int(val)
             
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    ser.close()
    if not 'Heart' in locals() :
        Heart=Ardu_Heart()
    else : top('심박수 : '+ str(Heart))
    return Heart

def Playtop(targ, tag='', siz=50, de=2000):
    pygame.mixer.init()
    pygame.mixer.music.load('assets/'+targ)
    pygame.mixer.music.play()
    top(tag, siz, de)

def Play(targ, dl=0):
    pygame.mixer.init()
    pygame.mixer.music.load('assets/'+targ)
    pygame.mixer.music.play()
    time.sleep(dl)

def arduino_rfid():
    Playtop("TAG.wav", 'Please Tag the reader.')
    ser=serial.Serial('COM8', 9600)
    if ser.readable():
        val = ser.readline()
        val=val.decode()[:len(val)-1]
    id, username = val.split(':')
    username=username.strip()
    #id='1803001'
    tid=int(id)
    #username='Kim Joo-Chan'
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    ser.close()
    Playtop("Recog.wav", 'Name : '+username+'\nID : '+ id)
    return tid, username

def upload_to_firebase(Pid=None, username=None, Belt_con=False, Shoes_con=False, Helmet_con=False, Tm_value=0.0, Hr_value=0, Al_value=0.0):
    cred = credentials.Certificate('lifevision-firebase.json')
    fire=firebase_admin.initialize_app(cred, {'porjectID': 'lifevision',})
    print('Uploading...', end=' ')
    db = firestore.client()
    t = datetime.today().strftime("%Y-%m-%d")
    doc_ref = db.collection(t).document(username)
    doc_ref.set({
        u'Belt': Belt_con,
        u'Helmet': Helmet_con,
        u'ID': Pid,
        u'Shoes': Shoes_con,
        u'Temp' : str(Tm_value)+'°C',
        u'HeartBeat' : Hr_value,
        u'Alcohol' : Al_value,
        u'Time': firestore.SERVER_TIMESTAMP
    })
    print('End.')
    firebase_admin.delete_app(fire)

def top(comment='', siz=50, dely=2000):
    top = Tk()
    Message(top, text=comment, font=("D2Coding", siz, "bold"), width=1080, padx=100, pady=200).pack()
    top.after(dely, top.destroy)
    top.mainloop()

if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = MyApp()
   sys.exit(app.exec_())