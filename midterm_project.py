import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Các hàm xử lý ảnh từ Notebook
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return np.round(gray).astype(np.uint8)

def gray2bin(img, threshold):
    return np.where(img > threshold, 1, 0)

def histogram(img):
    plt.hist(img.ravel(), bins=256, range=(0, 256))
    plt.show()

def median_filter(data, kernel_size):
    return gaussian_filter(data, kernel_size)

def mean_filter(data, kernel_size):
    return gaussian_filter(data,kernel_size/2)

def calculate_psnr_ssim(original, filtered):
    psnr_value = peak_signal_noise_ratio(original, filtered)
    ssim_value, _ = structural_similarity(original, filtered, full=True, win_size=3) 
    return psnr_value, ssim_value

# Tạo giao diện Tkinter
class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LAB1 - Image Processing")
        self.root.geometry("1000x600")

        # Tab LAB1
        self.tabControl = ttk.Notebook(root)
        self.lab1_tab = ttk.Frame(self.tabControl)
        self.tabControl.add(self.lab1_tab, text="LAB1")
        self.tabControl.pack(expand=1, fill="both")

        # Cột bên trái để hiển thị ảnh
        self.image_panel = tk.Label(self.lab1_tab)
        self.image_panel.grid(row=0, column=0, rowspan=15, padx=10, pady=10)

        # Cột bên phải với các nút chức năng
        # Chọn file ảnh
        self.img_label = tk.Label(self.lab1_tab, text="Choose an image:")
        self.img_label.grid(row=0, column=1, sticky="w", padx=5)
        
        self.img_button = tk.Button(self.lab1_tab, text="Browse", command=self.load_image)
        self.img_button.grid(row=0, column=2, padx=5, pady=5)

        # Chọn ngưỡng và kích thước bộ lọc
        self.threshold_label = tk.Label(self.lab1_tab, text="Threshold:")
        self.threshold_label.grid(row=1, column=1, sticky="w", padx=5)
        
        self.threshold_entry = tk.Entry(self.lab1_tab)
        self.threshold_entry.grid(row=1, column=2, padx=5, pady=5)
        
        self.filter_size_label = tk.Label(self.lab1_tab, text="Filter Size:")
        self.filter_size_label.grid(row=2, column=1, sticky="w", padx=5)
        
        self.filter_size_entry = tk.Entry(self.lab1_tab)
        self.filter_size_entry.grid(row=2, column=2, padx=5, pady=5)

        # Các nút chức năng
        self.rgb2gray_button = tk.Button(self.lab1_tab, text="Convert RGB to Gray", command=self.apply_rgb2gray)
        self.rgb2gray_button.grid(row=3, column=2, sticky="ew", padx=5, pady=5)

        self.gray2bin_button = tk.Button(self.lab1_tab, text="Convert Gray to Binary", command=self.apply_gray2bin)
        self.gray2bin_button.grid(row=4, column=2, sticky="ew", padx=5, pady=5)

        self.histogram_button = tk.Button(self.lab1_tab, text="Show Histogram", command=self.show_histogram)
        self.histogram_button.grid(row=5, column=2, sticky="ew", padx=5, pady=5)

        self.mean_filter_button = tk.Button(self.lab1_tab, text="Apply Mean Filter", command=self.apply_mean_filter)
        self.mean_filter_button.grid(row=6, column=2, sticky="ew", padx=5, pady=5)

        self.median_filter_button = tk.Button(self.lab1_tab, text="Apply Median Filter", command=self.apply_median_filter)
        self.median_filter_button.grid(row=7, column=2, sticky="ew", padx=5, pady=5)

        self.psnr_ssim_button = tk.Button(self.lab1_tab, text="Calculate PSNR and SSIM", command=self.calculate_metrics)
        self.psnr_ssim_button.grid(row=8, column=2, sticky="ew", padx=5, pady=5)

        self.extract_object_button = tk.Button(self.lab1_tab, text="Extract Object", command=self.extract_object)
        self.extract_object_button.grid(row=9, column=2, sticky="ew", padx=5, pady=5)
        
        self.reduce_brightness_button = tk.Button(self.lab1_tab, text="Reduce Brightness", command=self.reduce_brightness)
        self.reduce_brightness_button.grid(row=10, column=2, sticky="ew", padx=5, pady=5)
        
        self.sequence_filter_button = tk.Button(self.lab1_tab, text="Apply Sequential Filters", command=self.open_filter_window)
        self.sequence_filter_button.grid(row=11, column=2, sticky="ew", padx=5, pady=5)
        # Khởi tạo biến lưu ảnh
        self.original_img = None
        self.processed_img = None

    def open_filter_window(self):
        # Tạo cửa sổ mới
        filter_window = tk.Toplevel(self.root)
        filter_window.title("Choose Filter Sequence")
        filter_window.geometry("300x300")
        
        # Các nút lọc
        tk.Button(filter_window, text="Median -> Gaussian -> Mean", command=self.median_gaussian_mean).pack(pady=5)
        tk.Button(filter_window, text="Median -> Mean -> Gaussian", command=self.median_mean_gaussian).pack(pady=5)
        tk.Button(filter_window, text="Gaussian -> Median -> Mean", command=self.gaussian_median_mean).pack(pady=5)
        tk.Button(filter_window, text="Gaussian -> Mean -> Median", command=self.gaussian_mean_median).pack(pady=5)
        tk.Button(filter_window, text="Mean -> Median -> Gaussian", command=self.mean_median_gaussian).pack(pady=5)
        tk.Button(filter_window, text="Mean -> Gaussian -> Median", command=self.mean_gaussian_median).pack(pady=5)
        
        # Nút quay lại
        tk.Button(filter_window, text="Back", command=filter_window.destroy).pack(pady=20)

        # Thêm nút vào giao diện chính

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_img = Image.open(file_path).convert('RGB')
            img_display = ImageTk.PhotoImage(self.original_img)
            self.image_panel.config(image=img_display)
            self.image_panel.image = img_display

    def apply_rgb2gray(self):
        if self.original_img:
            img_array = np.array(self.original_img)
            gray_img = rgb2gray(img_array)
            self.show_image(gray_img)
            self.processed_img = gray_img

    def apply_gray2bin(self):
        if self.processed_img is not None:
            threshold = int(self.threshold_entry.get() or 128)
            binary_img = gray2bin(self.processed_img, threshold)
            self.show_image(binary_img * 255)

    def show_histogram(self):
        if self.processed_img is not None:
            histogram(self.processed_img)

    def apply_mean_filter(self):
        if self.processed_img is not None:
            kernel_size = int(self.filter_size_entry.get() or 3)
            filtered_img = mean_filter(self.processed_img, kernel_size)
            self.show_image(filtered_img)

    def apply_median_filter(self):
        if self.processed_img is not None:
            kernel_size = int(self.filter_size_entry.get() or 3)
            filtered_img = median_filter(self.processed_img, kernel_size)
            self.show_image(filtered_img)

    def calculate_metrics(self):
        if self.original_img is not None and self.processed_img is not None:
            processed_resized = np.resize(self.processed_img, np.array(self.original_img).shape)
            psnr, ssim = calculate_psnr_ssim(np.array(self.original_img), processed_resized)
            messagebox.showinfo("Metrics", f"PSNR: {psnr}\nSSIM: {ssim}")

    def show_image(self, img_array):
        img = Image.fromarray(np.uint8(img_array)).convert('RGB')
        img_display = ImageTk.PhotoImage(img)
        self.image_panel.config(image=img_display)
        self.image_panel.image = img_display
    
    # Thêm hàm tách vật thể
    def extract_object(self):
        if self.processed_img is not None:
            threshold = int(self.threshold_entry.get() or 128)
            binary_img = gray2bin(self.processed_img, threshold)
            # Nhân ảnh nhị phân với ảnh xám để chỉ giữ lại phần có vật thể
            object_img = binary_img * self.processed_img
            self.show_image(object_img)


    # Thêm hàm giảm độ sáng
    def reduce_brightness(self):
        if self.processed_img is not None:
            brightness_factor = 0.5  # Hệ số giảm sáng, có thể điều chỉnh
            reduced_brightness_img = (self.processed_img * brightness_factor).clip(0, 255)
            self.show_image(reduced_brightness_img)
    #
    def median_gaussian_mean(self):
        if self.processed_img is not None:
            # Lấy giá trị kernel từ người dùng
            kernel_size = int(self.filter_size_entry.get() or 3)
            # Bước 1: Áp dụng bộ lọc trung vị
            median_filtered_img = median_filter(self.processed_img, kernel_size)
            # Bước 2: Áp dụng bộ lọc Gaussian
            gaussian_filtered_img = gaussian_filter(median_filtered_img, kernel_size / 2)
            # Bước 3: Áp dụng bộ lọc trung bình
            mean_filtered_img = mean_filter(gaussian_filtered_img, kernel_size)
            # Hiển thị ảnh đã lọc
            self.show_image(mean_filtered_img)
    def median_mean_gaussian(self):
        if self.processed_img is not None:
            kernel_size = int(self.filter_size_entry.get() or 3)
            # Bước 1: Median filter
            median_filtered_img = median_filter(self.processed_img,kernel_size)
            # Bước 2: Mean filter
            mean_filtered_img = mean_filter(median_filtered_img,kernel_size)
            # Bước 3: Gaussian filter
            gaussian_filtered_img = gaussian_filter(mean_filtered_img, sigma=kernel_size / 2)
            
            # Hiển thị kết quả
            self.show_image(gaussian_filtered_img)
    def gaussian_median_mean(self):
        if self.processed_img is not None:
            kernel_size = int(self.filter_size_entry.get() or 3)
            
            # Bước 1: Gaussian filter
            gaussian_filtered_img = gaussian_filter(self.processed_img, sigma=kernel_size / 2)
            
            # Bước 2: Median filter
            median_filtered_img = median_filter(gaussian_filtered_img,kernel_size)
            
            # Bước 3: Mean filter
            mean_filtered_img = mean_filter(median_filtered_img,kernel_size)
            
            # Hiển thị kết quả
            self.show_image(mean_filtered_img)
    def gaussian_mean_median(self):
        if self.processed_img is not None:
            kernel_size = int(self.filter_size_entry.get() or 3)
            
            # Bước 1: Gaussian filter
            gaussian_filtered_img = gaussian_filter(self.processed_img, sigma=kernel_size / 2)
            
            # Bước 2: Mean filter
            mean_filtered_img = mean_filter(gaussian_filtered_img,kernel_size)
            
            # Bước 3: Median filter
            median_filtered_img = median_filter(mean_filtered_img,kernel_size)
            
            # Hiển thị kết quả
            self.show_image(median_filtered_img)
    def mean_median_gaussian(self):
        if self.processed_img is not None:
            kernel_size = int(self.filter_size_entry.get() or 3)
            
            # Bước 1: Mean filter
            mean_filtered_img = mean_filter(self.processed_img,kernel_size)
            
            # Bước 2: Median filter
            median_filtered_img = median_filter(mean_filtered_img,kernel_size)
            
            # Bước 3: Gaussian filter
            gaussian_filtered_img = gaussian_filter(median_filtered_img, sigma=kernel_size / 2)
            
            # Hiển thị kết quả
            self.show_image(gaussian_filtered_img)
    def mean_gaussian_median(self):
        if self.processed_img is not None:
            kernel_size = int(self.filter_size_entry.get() or 3)
            
            # Bước 1: Mean filter
            mean_filtered_img = mean_filter(self.processed_img,kernel_size)
            
            # Bước 2: Gaussian filter
            gaussian_filtered_img = gaussian_filter(mean_filtered_img, sigma=kernel_size / 2)
            
            # Bước 3: Median filter
            median_filtered_img = median_filter(gaussian_filtered_img,kernel_size)
            
            # Hiển thị kết quả
            self.show_image(median_filtered_img)


# Khởi chạy giao diện
root = tk.Tk()
app = ImageProcessingApp(root)
root.mainloop()
