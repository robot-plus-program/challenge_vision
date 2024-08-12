import os
import sys
from datetime import datetime

import natsort
from glob import glob
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.font as tkFont
import tkinter.ttk


class Labeller:
    def __init__(self):
        super(Labeller, self).__init__()

        self.ui = tk.Tk()

        self.ui.title("Center Point Labeller")
        self.ui.geometry("690x620")

        self.button_font = tkFont.Font(size=10)
        self.label_info_font = tkFont.Font(size=12)
        self.text_box_font = tkFont.Font(size=10)

        self.image_open = False
        self.path = tk.StringVar()
        self.path.set("/home/js/SynologyDrive/3. KETI_Research/workspace/inst_seg/inst_seg/centerpt/dataset/train")

        self.data = dict()
        self.data['image_path'] = ""
        self.data['label_path'] = ""

        self.set_labels()
        self.set_text_box()
        self.set_buttons()
        self.set_table()

        self.ui.focus_set()
        self.ui.bind('<Key>', self.on_key_press)

        self.ui.mainloop()

    def set_labels(self):
        self.Label1 = tk.Label(self.ui)
        self.Label2 = tk.Label(self.ui)
        self.Label3 = tk.Label(self.ui)
        self.Label4 = tk.Label(self.ui)
        self.Label5 = tk.Label(self.ui)
        self.Label6 = tk.Label(self.ui)

        self.Label1.place(x=10, y=90, width=520, height=520)
        self.Label1.configure(bg="#fff")
        self.Label1.bind("<Motion>", self.on_mouse_move)
        self.Label1.bind("<Button-1>", self.on_mouse_l_click)
        self.Label1.bind("<Button-3>", self.on_mouse_r_click)

        self.Label2.place(x=10, y=50, width=240, height=30)
        self.Label2.configure(bg="#fff")

        _Label3 = tk.Label(self.ui, text="Current", font=self.label_info_font)
        _Label3.place(x=260, y=50, height=30)
        self.Label3.place(x=330, y=50, width=70, height=30)
        self.Label3.configure(bg="#fff")

        _Label4 = tk.Label(self.ui, text="Total", font=self.label_info_font)
        _Label4.place(x=410, y=50, height=30)
        self.Label4.place(x=460, y=50, width=70, height=30)
        self.Label4.configure(bg="#fff")

        _Label5 = tk.Label(self.ui, text="Image size", font=self.label_info_font)
        _Label5.place(x=540, y=210, height=30)
        self.Label5.place(x=540, y=240, width=140, height=30)
        self.Label5.configure(bg="#fff")

        _Label6 = tk.Label(self.ui, text="Cursor", font=self.label_info_font)
        _Label6.place(x=540, y=270, height=30)
        self.Label6.place(x=540, y=300, width=140, height=30)
        self.Label6.configure(bg="#fff")

    def set_text_box(self):
        self.TxtBox1 = tk.ttk.Entry(self.ui)

        self.TxtBox1.place(x=10, y=10, width=520, height=30)
        self.TxtBox1.configure(textvariable=self.path,
                               font=self.text_box_font)

    def set_buttons(self):
        self.Button1 = tk.Button(self.ui)
        self.Button2 = tk.Button(self.ui)
        self.Button3 = tk.Button(self.ui)
        self.Button4 = tk.Button(self.ui)
        self.Button5 = tk.Button(self.ui)
        self.Button6 = tk.Button(self.ui)

        # Open
        self.Button1.place(x=540, y=10, width=140, height=30)
        self.Button1.configure(borderwidth="2",
                               text="Open (o)",
                               font=self.button_font,
                               command=self.click_button1)

        # Update
        self.Button2.place(x=540, y=50, width=140, height=30)
        self.Button2.configure(borderwidth="2",
                               text="Update (e)",
                               font=self.button_font,
                               command=self.click_button2)

        # Before
        self.Button3.place(x=540, y=90, width=65, height=30)
        self.Button3.configure(borderwidth="2",
                               text="< (a)",
                               font=self.button_font,
                               command=self.click_button3)

        # After
        self.Button4.place(x=616, y=90, width=65, height=30)
        self.Button4.configure(borderwidth="2",
                               text="> (d)",
                               font=self.button_font,
                               command=self.click_button4)

        # Save label
        self.Button5.place(x=540, y=130, width=140, height=30)
        self.Button5.configure(borderwidth="2",
                               text="Save (s)",
                               font=self.button_font,
                               command=self.click_button5)

        # Remove image & label
        self.Button6.place(x=540, y=170, width=140, height=30)
        self.Button6.configure(borderwidth="2",
                               text="Remove (r)",
                               font=self.button_font,
                               command=self.click_button6)

    def set_table(self):
        _Table1 = tk.Label(self.ui, text="Label", font=self.label_info_font)
        _Table1.place(x=540, y=330, height=30)
        self.Table1 = tk.ttk.Treeview(self.ui,
                                      columns=["axis", "value"],
                                      displaycolumns=["axis", "value"])
        self.Table1.place(x=540, y=360, width=140, height=60)
        self.Table1.column("axis", width=60, anchor="center")
        self.Table1.heading("axis", text="Axis", anchor="center")
        self.Table1.column("value", width=60, anchor="center")
        self.Table1.heading("value", text="Value", anchor="center")
        self.Table1["show"] = "headings"

    def click_button1(self):
        try:
            path = self.path.get()
            self.data['image_path'] = f"{path}/image"
            self.data['label_path'] = f"{path}/label"

            self.data['image_list'] = natsort.natsorted(os.listdir(self.data['image_path']))
            self.data['label_list'] = natsort.natsorted(os.listdir(self.data['label_path']))
            self.data['num_data'] = len(self.data['image_list'])

            self.data['data_list'] = []
            for i in range(self.data['num_data']):
                self.data['data_list'].append(self.load_data(i))

            try:
                if self.data['current_index']:
                    pass
                elif self.data['current_index'] > (self.data['num_data'] - 1):
                    self.data['current_index'] -= 1
            except:
                self.data['current_index'] = 0

            self.update_ui()

            self.image_open = True

        except Exception as e:
            print(e)

    def click_button2(self):
        pass

    def click_button3(self):
        self.data['current_index'] -= 1

        if self.data['current_index'] < 0:
            self.data['current_index'] = self.data['num_data'] - 1

        self.update_ui()

    def click_button4(self):
        self.data['current_index'] += 1

        if self.data['current_index'] > (self.data['num_data'] - 1):
            self.data['current_index'] = 0

        self.update_ui()

    def click_button5(self):
        d = self.get_cur_data()
        write_txt(f"{self.data['label_path']}/{d['fname']}.txt", d['label'])

    def click_button6(self):
        pass

    def update_ui(self):
        d = self.get_cur_data()
        self.update_ui_img(d)
        self.update_text_label(d)
        self.update_table(d, self.Table1)

        write_txt(f"{self.data['label_path']}/{d['fname']}.txt", d['label'])

    def load_data(self, index):
        data = dict()

        image_fname = self.data['image_list'][index]
        fname = image_fname.split('.')[0]

        try:
            label_fname = self.data['label_list'][index]
        except:
            label_fname = f"{fname}.txt"

        image = cv2.imread(f"{self.data['image_path']}/{image_fname}")[:, :, ::-1]
        label = open_txt(f"{self.data['label_path']}/{label_fname}")

        data['index'] = index
        data['fname'] = fname
        data['image'] = image
        data['label'] = label
        data['shape'] = image.shape
        data['dtype'] = image.dtype

        return data

    def update_ui_img(self, d):
        if d['label']:
            image = np.array(d['image'])
            image = cv2.circle(image, d['label'], radius=6, color=(0, 0, 255), thickness=-1)
            image = cv2.circle(image, d['label'], radius=4, color=(255, 0, 0), thickness=-1)
            image = cv2.circle(image, d['label'], radius=2, color=(0, 255, 0), thickness=-1)
        else:
            image = d['image']
        self.show_ui_img(image, self.Label1)

    def update_text_label(self, d):
        self.Label2.configure(text=d['fname'], font=self.label_info_font,
                              anchor='w', justify='left')
        self.Label3.configure(text=d['index']+1, font=self.label_info_font)

        self.Label4.configure(text=self.data['num_data'], font=self.label_info_font)
        self.Label5.configure(text=str(d['shape']))

    def update_table(self, d, table):
        table.delete(*table.get_children())

        if d['label']:
            table.insert("", "end", text="", values=("x", d['label'][0]), iid=0)
            table.insert("", "end", text="", values=("y", d['label'][1]), iid=1)
        else:
            table.insert("", "end", text="", values=("x", ""), iid=0)
            table.insert("", "end", text="", values=("y", ""), iid=1)

    def on_mouse_move(self, event):
        if self.image_open:
            x, y = event.x, event.y

            d = self.get_cur_data()

            x /= self.Label1.winfo_width()
            y /= self.Label1.winfo_height()

            x *= d['shape'][1]
            y *= d['shape'][0]

            x = int(x)
            y = int(y)

            self.data['cursor'] = [x, y]
            cursor_str = f"x: {x} | y: {y}"
            self.Label6.configure(text=cursor_str)

    def on_mouse_l_click(self, event):
        if self.image_open:
            d = self.get_cur_data()
            d['label'] = self.data['cursor']

            self.update_ui()

    def on_mouse_r_click(self, event):
        if self.image_open:
            d = self.get_cur_data()
            d['label'] = False

            self.update_ui()

    def on_key_press(self, event):
        k = event.keysym
        if k == 'o':
            self.click_button1()
        elif k == 'e':
            self.click_button2()
        elif k == 'a':
            self.click_button3()
        elif k == 'd':
            self.click_button4()
        elif k == 's':
            self.click_button5()
        elif k == 'r':
            self.click_button6()

    @staticmethod
    def show_ui_img(image, label):
        try:
            image = Image.fromarray(cv2.resize(image, (label.winfo_width(), label.winfo_height())))
            image = ImageTk.PhotoImage(image)

            if label is None:
                label = tk.Label(image=image)
                label.image = image
                label.pack(side="left")
            else:
                label.configure(image=image)
                label.image = image
        except:
            pass

    def get_cur_data(self):
        return self.data['data_list'][self.data['current_index']]

def open_txt(path):
    try:
        with open(path, "r") as f:
            data = f.readline()
        if data:
            data = [int(i) for i in data.split(',')]
            return data
        else:
            return False
    except:
        return False

def write_txt(path, content):
    if content:
        c = f"{content[0]}, {content[1]}"
    else:
        c = ""

    try:
        with open(path, "w") as f:
            f.write(c)
    except:
        pass

if __name__ == "__main__":
    labeller = Labeller()
