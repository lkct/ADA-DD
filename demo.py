from tkinter import *
from tkinter.filedialog import *
from PIL import Image, ImageTk
import hashlib
import time
import cv2
import ddr.dev.col as col
import ddr.src.main as smain

LOG_LINE_NUM = 0


def resize(w, h, w_box, h_box, pil_image):
    f1 = 1.0*w_box/w  # 1.0 forces float division in Python2
    f2 = 1.0*h_box/h
    factor = min([f1, f2])
    # print(f1, f2, factor) # test
    # use best down-sizing filter
    width = int(w*factor)
    height = int(h*factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)


class MY_GUI():
    def __init__(self, init_window_name):
        self.init_window_name = init_window_name

    def set_init_window(self):
        self.init_window_name.title("ADA-DD_DDR ver")
        self.init_window_name.geometry('250x50+10+10')
        self.init_window_name["bg"] = "pink"
        self.init_window_name.attributes("-alpha", 0.9)

        # self.init_data_label = Label(self.init_window_name, text="to recog")
        # self.init_data_label.grid(row=0, column=0)
        # self.result_data_label = Label(self.init_window_name, text="result")
        # self.result_data_label.grid(row=0, column=12)
        # self.log_label = Label(self.init_window_name, text="Log")
        # self.log_label.grid(row=12, column=0)

        # loa = Image.open('gui/8.jpg')
        # w, h = loa.size
        # loa_resized = resize(w, h, 1000, 500, loa)
        # load = ImageTk.PhotoImage(loa_resized)
        # self.loadLabel = Label(self.init_window_name, image=load)
        # self.loadLabel.grid(row=1, column=0, columnspan=3)
        #load = PhotoImage(file="F:\\PKU\\workplace\\gui\\8.jpg")
        #loadLabel = Label(self.init_window_name, image=load)
        #loadLabel.grid(row=1, column=0)

        # self.init_data_Text = Text(self.init_window_name, width=67, height=35)
        #self.init_data_Text.grid(row=1, column=0, rowspan=10, columnspan=10)
        # self.result_data_Text = Text(self.init_window_name, width=70, height=49)
        #self.result_data_Text.grid(row=1, column=12, rowspan=15, columnspan=10)
        # self.log_data_Text = Text(
        #     self.init_window_name, width=66, height=9)
        # self.log_data_Text.grid(row=13, column=0, columnspan=10)

        self.img_choose_button = Button(
            self.init_window_name, text="Choose a image", bg="lightblue", width=10, command=self.img_choose_and_recog)
        self.img_choose_button.grid(row=0, column=0)
        # self.str_trans_to_md5_button = Button(
        #     self.init_window_name, text="start", bg="lightblue", width=10, command=None)
        # self.str_trans_to_md5_button.grid(row=1, column=3)

    def img_choose_and_recog(self):
        filename = askopenfilename(filetypes=[("jpg file", "jpg")])
        # filename = '/mnt/ssd/ADA-DD/dev/10.jpg'
        print filename
        img = cv2.imread(filename)
        img = col.proc(img)
        anno, lab, prob = smain.recog(img, 'result.jpg')
        if anno.shape[0] != 0:
            anno[:, 2] += anno[:, 0]
            anno[:, 3] += anno[:, 1]
        annofile = open('result.txt', 'w')
        annofile.write('x1\ty1\tx2\ty2\tlabel\tconfidence\n')
        for i in range(lab.size):
            annofile.write('%d\t%d\t%d\t%d\t%d\t%f\n' % (
                anno[i, 0], anno[i, 1], anno[i, 2], anno[i, 3], lab[i], prob[i]))
        annofile.close()
        print 'done'

    def str_trans_to_md5(self):
        src = self.init_data_Text.get(
            1.0, END).strip().replace("\n", "").encode()
        #print("src =",src)
        if src:
            try:
                myMd5 = hashlib.md5()
                myMd5.update(src)
                myMd5_Digest = myMd5.hexdigest()

                self.result_data_Text.delete(1.0, END)
                self.result_data_Text.insert(1.0, myMd5_Digest)
                self.write_log_to_Text("INFO:str_trans_to_md5 success")
            except:
                self.result_data_Text.delete(1.0, END)
                self.result_data_Text.insert(
                    1.0, "ERROR:str_trans_to_md5 failed")
        else:
            self.write_log_to_Text("ERROR:str_trans_to_md5 failed")

    def get_current_time(self):
        current_time = time.strftime(
            '%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        return current_time

    def write_log_to_Text(self, logmsg):
        global LOG_LINE_NUM
        current_time = self.get_current_time()
        logmsg_in = str(current_time) + " " + str(logmsg) + "\n"
        if LOG_LINE_NUM <= 7:
            self.log_data_Text.insert(END, logmsg_in)
            LOG_LINE_NUM = LOG_LINE_NUM + 1
        else:
            self.log_data_Text.delete(1.0, 2.0)
            self.log_data_Text.insert(END, logmsg_in)


def gui_start():
    init_window = Tk()
    ZMJ_PORTAL = MY_GUI(init_window)

    ZMJ_PORTAL.set_init_window()

    init_window.mainloop()


gui_start()
