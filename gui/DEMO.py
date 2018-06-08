from tkinter import *
from tkinter.filedialog import *
from PIL import Image, ImageTk
import hashlib
import time

LOG_LINE_NUM = 0

def resize(w, h, w_box, h_box, pil_image):    
  f1 = 1.0*w_box/w # 1.0 forces float division in Python2    
  f2 = 1.0*h_box/h    
  factor = min([f1, f2])    
  #print(f1, f2, factor) # test    
  # use best down-sizing filter    
  width = int(w*factor)    
  height = int(h*factor)    
  return pil_image.resize((width, height), Image.ANTIALIAS) 

class MY_GUI():
    def __init__(self,init_window_name):
        self.init_window_name = init_window_name


    #设置窗口
    def set_init_window(self):
        self.init_window_name.title("ADA-DD_DDR版")           #窗口名
        self.init_window_name.geometry('1068x681+10+10')
        self.init_window_name["bg"] = "pink"
        self.init_window_name.attributes("-alpha",0.9)
        #标签
        self.init_data_label = Label(self.init_window_name, text="待识别图片")
        self.init_data_label.grid(row=0, column=0)
        self.result_data_label = Label(self.init_window_name, text="识别结果")
        self.result_data_label.grid(row=0, column=12)
        self.log_label = Label(self.init_window_name, text="日志")
        self.log_label.grid(row=12, column=0)
        #图片
        loa = Image.open('8.jpg')
        w, h = loa.size
        loa_resized = resize(w, h, 1000, 500, loa)
        load = ImageTk.PhotoImage(loa_resized)
        self.loadLabel = Label(self.init_window_name, image=load)
        self.loadLabel.grid(row=1, column=0, columnspan=3)
        #load = PhotoImage(file="F:\\PKU\\workplace\\gui\\8.jpg")
        #loadLabel = Label(self.init_window_name, image=load)
        #loadLabel.grid(row=1, column=0)
        #文本框
        #self.init_data_Text = Text(self.init_window_name, width=67, height=35)  #原始数据录入框
        #self.init_data_Text.grid(row=1, column=0, rowspan=10, columnspan=10)
        #self.result_data_Text = Text(self.init_window_name, width=70, height=49)  #处理结果展示
        #self.result_data_Text.grid(row=1, column=12, rowspan=15, columnspan=10)
        self.log_data_Text = Text(self.init_window_name, width=66, height=9)  # 日志框
        self.log_data_Text.grid(row=13, column=0, columnspan=10)
        #按钮
        self.img_choose_and_show_button = Button(self.init_window_name, text="选择图片", bg="lightblue", width=10, command=self.img_choose_and_show)
        self.img_choose_and_show_button.grid(row=0, column=3)
        self.str_trans_to_md5_button = Button(self.init_window_name, text="开始识别", bg="lightblue", width=10,command="这里缺个调用的识别函数")  # 调用内部方法  加()为直接调用
        self.str_trans_to_md5_button.grid(row=1, column=3)

    #选择图片
    def img_choose_and_show(self):
        filename=askopenfilename(filetypes=[("jpg格式","jpg")])
        load = ImageTk.PhotoImage(file=filename)
        loadLabel = Label(self.init_window_name, image=load)
        loadLabel.grid(row=1, column=0)

    #功能函数
    def str_trans_to_md5(self):
        src = self.init_data_Text.get(1.0,END).strip().replace("\n","").encode()
        #print("src =",src)
        if src:
            try:
                myMd5 = hashlib.md5()
                myMd5.update(src)
                myMd5_Digest = myMd5.hexdigest()
                #输出到界面
                self.result_data_Text.delete(1.0,END)
                self.result_data_Text.insert(1.0,myMd5_Digest)
                self.write_log_to_Text("INFO:str_trans_to_md5 success")
            except:
                self.result_data_Text.delete(1.0,END)
                self.result_data_Text.insert(1.0,"字符串转MD5失败")
        else:
            self.write_log_to_Text("ERROR:str_trans_to_md5 failed")


    #获取当前时间
    def get_current_time(self):
        current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        return current_time


    #日志动态打印
    def write_log_to_Text(self,logmsg):
        global LOG_LINE_NUM
        current_time = self.get_current_time()
        logmsg_in = str(current_time) +" " + str(logmsg) + "\n"      #换行
        if LOG_LINE_NUM <= 7:
            self.log_data_Text.insert(END, logmsg_in)
            LOG_LINE_NUM = LOG_LINE_NUM + 1
        else:
            self.log_data_Text.delete(1.0,2.0)
            self.log_data_Text.insert(END, logmsg_in)


def gui_start():
    init_window = Tk()              #实例化出一个父窗口
    ZMJ_PORTAL = MY_GUI(init_window)
    # 设置根窗口默认属性
    ZMJ_PORTAL.set_init_window()

    init_window.mainloop()          #父窗口进入事件循环，可以理解为保持窗口运行，否则界面不展示


gui_start()
