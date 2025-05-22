import tkinter as tk
from tkinter import ttk, messagebox
import os
from PIL import Image, ImageTk
from utils.test_single import run
from environments.warshall import warshall
from utils.utils import resource_path

user_list = ["user1", "user2"]

password_list = {"user1": "123456",
                 "user2": "234567"}


def set_background(root, image_path):
    bg_image = Image.open(image_path)
    bg_image_tk = ImageTk.PhotoImage(bg_image)

    background_label = tk.Label(root, image=bg_image_tk)
    background_label.image = bg_image_tk  # 保存引用
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    return background_label  # 可用于后续调整


def backend_login(user, password):
    return password == password_list[user]  # 密码


def backend_run_transformation(module_num, initial_coords, target_coords):
    run(module_num, initial_coords, target_coords)
    fig_dir = []
    files = os.listdir(resource_path('app_plot'))
    files.sort(key=lambda x: int(x.split('.')[0]))
    for filename in files:
        fig_dir.append(os.path.join(resource_path('app_plot'), filename))
    return fig_dir


class LoginWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("多模块卫星自重构路径规划可视化系统")
        self.root.geometry("800x800")
        self.root.resizable(False, False)

        set_background(self.root, resource_path("fig/login.jpg"))

        tk.Label(root, text="多模块卫星自重构路径规划可视化系统", font=("Microsoft YaHei", 16, "bold")).pack(pady=10)

        tk.Label(root, text="用户名：").pack()
        self.username = ttk.Combobox(root, values=user_list, state="readonly")
        self.username.pack()

        tk.Label(root, text="密码：").pack()
        self.password = tk.Entry(root, show="*")
        self.password.pack()

        btn_frame = tk.Frame(root)  # 在屏幕上显示一个矩形区域，多用来作为容器
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="登录", command=self.login).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="取消", command=root.quit).grid(row=0, column=1, padx=5)

    def login(self):
        user = self.username.get()
        pwd = self.password.get()
        if backend_login(user, pwd):
            self.root.destroy()
            MainApp()
        else:
            messagebox.showerror("错误", "用户名或密码错误")


class MainApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("多模块卫星自重构路径规划可视化系统")
        self.root.geometry("1400x800")

        self.module_count = tk.StringVar()
        self.coord_entries_init = []
        self.coord_entries_target = []
        self.images = []

        self.setup_menu()
        self.root.mainloop()

    def setup_menu(self):
        self.menu_frame = tk.Frame(self.root)
        self.menu_frame.pack(pady=10)

        # 模块个数
        tk.Label(self.menu_frame, text="模块个数：").grid(row=0, column=0)
        self.module_selector = ttk.Combobox(self.menu_frame, values=["4", "6"], state="readonly")
        self.module_selector.grid(row=0, column=1)
        self.module_selector.bind("<<ComboboxSelected>>", self.on_module_count_selected)

        # 初始构型、目标构型按钮
        self.init_button = tk.Button(self.menu_frame, text="重置初始构型", state="disabled",
                                     command=self.show_initial_input)
        self.init_button.grid(row=0, column=2, padx=10)

        self.target_button = tk.Button(self.menu_frame, text="重置目标构型", state="disabled",
                                       command=self.show_target_input)
        self.target_button.grid(row=0, column=3, padx=10)

        self.trans_button = tk.Button(self.menu_frame, text="计算变构方案", state="disabled",
                                      command=self.run_transformation)
        self.trans_button.grid(row=0, column=4, padx=10)

        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack(pady=20)

        # 创建一个水平布局容器
        self.config_frame = tk.Frame(self.root)
        self.config_frame.pack(pady=10, fill="both", expand=True)

        # 左侧：初始构型区域
        self.init_frame = tk.Frame(self.config_frame)
        self.init_frame.grid(row=0, column=0, padx=(200, 40), sticky="nw")  # 偏左一些

        # 中间空白（可选）
        self.middle_spacer = tk.Frame(self.config_frame, width=250)
        self.middle_spacer.grid(row=0, column=1)

        # 右侧：目标构型区域
        self.target_frame = tk.Frame(self.config_frame)
        self.target_frame.grid(row=0, column=2, padx=(40, 200), sticky="ne")  # 偏右一些

        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack()

    def on_module_count_selected(self, event):
        self.module_count = int(self.module_selector.get())
        self.init_button.config(state="normal")
        self.target_button.config(state="normal")
        self.clear_frame(self.input_frame, None)
        self.coord_entries_init.clear()
        self.coord_entries_target.clear()
        self.trans_button.config(state="disabled")

    def show_initial_input(self):
        # 清空原内容 & 清除旧 Entry 引用
        self.clear_frame(self.init_frame, self.coord_entries_init)

        # 添加标题行
        tk.Label(self.init_frame, text="初始构型").grid(row=0, column=0, padx=5, pady=5)
        tk.Label(self.init_frame, text="模块").grid(row=1, column=0, padx=5, pady=5)
        tk.Label(self.init_frame, text="X").grid(row=1, column=1, padx=5, pady=5)
        tk.Label(self.init_frame, text="Y").grid(row=1, column=2, padx=5, pady=5)
        tk.Label(self.init_frame, text="Z").grid(row=1, column=3, padx=5, pady=5)

        # 为每个模块添加输入行
        for i in range(self.module_count):
            tk.Label(self.init_frame, text=f"模块{i + 1}").grid(row=i + 2, column=0, padx=5, pady=5)
            row_entries = []
            for j in range(3):
                entry = tk.Entry(self.init_frame, width=10)

                if i == 0:
                    entry.insert(0, "0")  # 设置只读默认值
                    entry.config(state="readonly")

                entry.grid(row=i + 2, column=j + 1, padx=5, pady=5)
                row_entries.append(entry)

            self.coord_entries_init.append(row_entries)

    def show_target_input(self):
        self.clear_frame(self.target_frame, self.coord_entries_target)
        # 添加标题行
        tk.Label(self.target_frame, text="目标构型").grid(row=0, column=0, padx=5, pady=5)
        tk.Label(self.target_frame, text="模块").grid(row=1, column=0, padx=5, pady=5)
        tk.Label(self.target_frame, text="X").grid(row=1, column=1, padx=5, pady=5)
        tk.Label(self.target_frame, text="Y").grid(row=1, column=2, padx=5, pady=5)
        tk.Label(self.target_frame, text="Z").grid(row=1, column=3, padx=5, pady=5)

        # 为每个模块添加输入行
        for i in range(self.module_count):
            tk.Label(self.target_frame, text=f"模块{i + 1}").grid(row=i + 2, column=0, padx=5, pady=5)
            row_entries = []
            for j in range(3):
                entry = tk.Entry(self.target_frame, width=10)

                if i == 0:
                    entry.insert(0, "0")  # 设置只读默认值
                    entry.config(state="readonly")

                entry.grid(row=i + 2, column=j + 1, padx=5, pady=5)
                row_entries.append(entry)

            self.coord_entries_target.append(row_entries)

        self.trans_button.config(state="normal")

    def run_transformation(self):
        initial_coords = self.read_coords(self.coord_entries_init)
        target_coords = self.read_coords(self.coord_entries_target)
        module_num = self.module_count
        if not initial_coords or not target_coords:
            messagebox.showerror("错误", "请填写完整并确保输入合法且构型联通")
            return

        image_paths = backend_run_transformation(module_num, initial_coords, target_coords)
        self.show_transformation_images(image_paths)

    def read_coords(self, coord_list):
        result = []
        for row in coord_list:
            row_coords = []
            for entry in row:
                val = entry.get()
                if not val.lstrip("-").isdigit():
                    return None
                row_coords.append(int(val))
            result.append(row_coords)
        if warshall(result) != 0:
            return None
        return result

    def show_transformation_images(self, image_list, images_per_row=4):
        # 清空旧区域
        self.clear_frame(self.image_frame, None)
        self.images.clear()
        config_label = tk.Label(self.image_frame, text='变构方案')
        config_label.pack(anchor="nw", padx=10, pady=10)
        # 创建滚动区域
        canvas = tk.Canvas(self.image_frame, width=1000, height=500)
        scrollbar = tk.Scrollbar(self.image_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 加载并显示图片
        self.image_refs = []  # 防止图像被垃圾回收
        for idx, img_path in enumerate(image_list):
            row = idx // images_per_row
            col = idx % images_per_row

            # 加载图片（你可能需要 resize）
            pil_img = Image.open(img_path).resize((200, 200))  # 可调整尺寸
            tk_img = ImageTk.PhotoImage(pil_img)
            self.image_refs.append(tk_img)

            # 图片标签
            img_label = tk.Label(scrollable_frame, image=tk_img)
            img_label.grid(row=row * 2, column=col, padx=10, pady=5)

            # 下方文字标签
            if idx == 0:
                text_label = tk.Label(scrollable_frame, text=f"initial config")
            elif idx == len(image_list)-1:
                text_label = tk.Label(scrollable_frame, text=f"goal config")
            else:
                text_label = tk.Label(scrollable_frame, text=f"step {idx}")

            text_label.grid(row=row * 2 + 1, column=col)

    def clear_frame(self, frame, clear_list=None):
        for widget in frame.winfo_children():
            widget.destroy()
        if clear_list is not None:
            clear_list.clear()


def boot():
    root = tk.Tk()
    app = LoginWindow(root)
    root.mainloop()
