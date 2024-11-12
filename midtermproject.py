import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from skimage import exposure
import cv2

# Các hàm xử lý ảnh từ Notebook lab 1
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
# Các hàm xử lý ảnh từ Notebook lab 2
def prewitt_edge_detection(image):
    img = image
    img = img.convert("L")
    img = np.array(img, dtype=np.float32) 
    gx = np.zeros((len(img),len(img[0])))
    gy = np.zeros((len(img),len(img[0])))
    img_final = np.zeros((len(img),len(img[0])))
    for i in range(1, len(img) - 1): # Chiều cao (đại diện cho y)
        for j in range(1, len(img[0]) - 1): # Chiều rộng (đại diện cho x)
            gx[i, j] = (img[i - 1, j - 1] + img[i, j - 1] + img[i + 1, j - 1]) - (img[i - 1, j + 1] + img[i, j + 1] + img[i + 1, j + 1]) 
            gy[i, j] = (img[i - 1, j - 1] + img[i - 1, j] + img[i - 1, j + 1]) - (img[i + 1, j - 1] + img[i + 1, j] + img[i + 1, j + 1]) 
            img_final[i, j] =min(255, np.sqrt(gx[i, j]**2 + gy[i, j]**2))
    return gx,gy,img_final

# Tạo giao diện Tkinter
class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GUI - Image Processing")
        self.root.geometry("1000x600")
        self.display_size = (300, 300)

        # Tab LAB1
        self.tabControl = ttk.Notebook(root)
        self.lab1_tab = ttk.Frame(self.tabControl)
        self.tabControl.add(self.lab1_tab, text="LAB1")
        self.tabControl.pack(expand=1, fill="both")
        # Tab LAB2
        self.lab2_tab = ttk.Frame(self.tabControl)
        self.tabControl.add(self.lab2_tab, text="LAB2")
        self.tabControl.pack(expand=1, fill="both")


        #############################################   LAB1    #############################################
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
        
        self.gauss_filter_button = tk.Button(self.lab1_tab, text="Apply Gauss Filter", command=self.apply_gauss_filter)
        self.gauss_filter_button.grid(row=12, column=2, sticky="ew", padx=5, pady=5)
        
        self.histogram_equalization_button = tk.Button(self.lab1_tab, text="Histogram equalization", command=self.apply_histogram_equalization)
        self.histogram_equalization_button.grid(row=13, column=2, sticky="ew", padx=5, pady=5)

   
        self.salt_pepper_noise_button = tk.Button(self.lab1_tab, text="Add Salt and Pepper Noise", command=self.add_salt_pepper_noise)
        self.salt_pepper_noise_button.grid(row=14, column=2, sticky="ew", padx=5, pady=5)

        
        self.gaussian_noise_button = tk.Button(self.lab1_tab, text="Add Gaussian Noise", command=self.add_gaussian_noise)
        self.gaussian_noise_button.grid(row=15, column=2, sticky="ew", padx=5, pady=5)
    ## biến lab 1
        self.original_img = None
        self.processed_img = None
    ## biến lab 2
        self.image_original_lab2 = None
    #################################################   LAB2   #################################################
    # Cột bên trái để hiển thị ảnh
        self.image_panel_lab2 = tk.Label(self.lab2_tab)
        self.image_panel_lab2.grid(row=0, column=0, rowspan=15, padx=10, pady=10)

        # Cột bên phải với các nút chức năng
        # Chọn file ảnh
        self.img_label = tk.Label(self.lab2_tab, text="Choose an image:")
        self.img_label.grid(row=0, column=1, sticky="w", padx=5)
        
        self.img_button = tk.Button(self.lab2_tab, text="Browse", command=self.load_image)
        self.img_button.grid(row=0, column=2, padx=5, pady=5)

        self.prewitt_edge_detection_button = tk.Button(self.lab2_tab, text="prewitt edge detection", command=self.apply_prewitt_edge_detection)
        self.prewitt_edge_detection_button.grid(row=3, column=2, sticky="ew", padx=5, pady=5)
        
        self.prewitt_edge_detection_button = tk.Button(self.lab2_tab, text="Add noise and Detect Canny", command=self.apply_add_noise_and_detect_canny)
        self.prewitt_edge_detection_button.grid(row=4, column=2, sticky="ew", padx=5, pady=5)


    ################################################################################################################################

    #Các Hàm con sử dụng trong các lab
    ############################################   LAB1    ############################################
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
            # Resize image to fit within 300x300 pixels for display
            display_size = (300, 300)
            img_resized = self.original_img.resize(display_size, Image.LANCZOS)
            img_display = ImageTk.PhotoImage(img_resized)
            # Load image based on the active tab
            current_tab_id = self.tabControl.select()
            if current_tab_id == str(self.lab1_tab):
                self.original_img = img_resized
                self.image_panel.config(image=img_display)
                self.image_panel.image = img_display
            elif current_tab_id == str(self.lab2_tab):
                self.image_original_lab2 = img_resized
                self.image_panel_lab2.config(image=img_display)
                self.image_panel_lab2.image = img_display
            

    def apply_rgb2gray(self):
        if self.original_img:
            img_array = np.array(self.original_img)
            gray_img = rgb2gray(img_array)
            gray_img_resized = self.resize_to_display(gray_img)
            self.processed_img = gray_img_resized
            self.show_image(gray_img_resized)

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
            filtered_img = np.array(self.original_img)
            filtered_img = mean_filter(filtered_img, kernel_size)
            self.show_image(filtered_img)

    def apply_median_filter(self):
        if self.processed_img is not None:
            kernel_size = int(self.filter_size_entry.get() or 3)
            filtered_img = np.array(self.original_img)
            filtered_img = median_filter(filtered_img, kernel_size)
            self.show_image(filtered_img)

    def calculate_metrics(self):
        if self.original_img is not None and self.processed_img is not None:
            processed_resized = np.resize(self.processed_img, np.array(self.original_img).shape)
            psnr, ssim = calculate_psnr_ssim(np.array(self.original_img), processed_resized)
            messagebox.showinfo("Metrics", f"PSNR: {psnr}\nSSIM: {ssim}")

    def resize_to_display(self, img_array):
        img = Image.fromarray(img_array.astype(np.uint8))
        resized_img = img.resize(self.display_size, Image.LANCZOS)
        return np.array(resized_img)

    def show_image(self, img_array):
        img_array_resized = self.resize_to_display(img_array)
        img = Image.fromarray(np.uint8(img_array_resized)).convert('L')
        img_display = ImageTk.PhotoImage(img)

        current_tab_id = self.tabControl.select()

        ####### chọn tab hiển thị #######
        if current_tab_id == str(self.lab1_tab):
            self.image_panel.config(image=img_display)
            self.image_panel.image = img_display
        elif current_tab_id == str(self.lab2_tab):
            self.image_panel_lab2.config(image=img_display)
            self.image_panel_lab2.image = img_display
    
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

    def apply_gauss_filter(self):
        if self.processed_img is not None:
            kernel_size = int(self.filter_size_entry.get() or 3)
            gauss_img = gaussian_filter(self.processed_img, sigma=kernel_size / 2)
            self.show_image(gauss_img) 
     
    def apply_histogram_equalization(self):                  
        if self.processed_img is not None:
            # Ensure the image is in the correct format for histogram equalization
            img_float = self.processed_img / 255.0  # Normalize to [0, 1] range
            equalized_img = exposure.equalize_hist(img_float) * 255  # Apply equalization and scale back to [0, 255]
            self.show_image(equalized_img.astype(np.uint8))  # Convert to uint8 for display
    def add_salt_pepper_noise(self):
        if self.processed_img is not None:
            # Thêm nhiễu hạt tiêu
            noisy_img = random_noise(self.processed_img, mode='s&p', amount=0.05)  # mức độ nhiễu
            noisy_img = (noisy_img * 255).astype(np.uint8)  # chuyển đổi sang uint8
            self.processed_img = noisy_img  # cập nhật processed_img với ảnh có nhiễu
            self.show_image(noisy_img)  # hiển thị ảnh

    def add_gaussian_noise(self):
        if self.processed_img is not None:
            # Thêm nhiễu Gaussian
            noisy_img = random_noise(self.processed_img, mode='gaussian', var=0.01)  # phương sai của nhiễu
            noisy_img = (noisy_img * 255).astype(np.uint8)  # chuyển đổi sang uint8
            self.processed_img = noisy_img  # cập nhật processed_img với ảnh có nhiễu
            self.show_image(noisy_img)  # hiển thị ảnh
################################################################################################################################
######################################################### LAB 2 ###############################################################

    def apply_prewitt_edge_detection(self):
        if self.image_original_lab2 is not None:
            # Áp dụng bộ lọc Prewitt để phát hiện cạnh
            gx, gy, prewitt_edges = prewitt_edge_detection(self.image_original_lab2)
            # Hiển thị kết quả ảnh cạnh đã phát hiện bằng Prewitt
            self.show_image(prewitt_edges)
    def apply_add_noise_and_detect_canny(self):
        if self.image_original_lab2 is not None:
            # Thêm nhiễu muối tiêu
            img_arr = np.array(self.image_original_lab2)
            sp_img = random_noise(img_arr, mode='s&p', amount=0.03)
            # Thêm nhiễu Gauss
            gauss_img = random_noise(sp_img, mode='gaussian', mean=0, var=0.02)
            img_eq = np.rint(255 * exposure.equalize_hist(gauss_img, nbins=256)).astype(np.uint8)
            # Loại bỏ nhiễu bằng bộ lọc trung bình
            remove_noise1 = mean_filter(img_eq,3)
            remove_noise2 = gaussian_filter(remove_noise1,sigma=2, radius=2)
            # Loại bỏ nhiễu bằng bộ lọc Gauss
            remove_noise2 = gaussian_filter(remove_noise1, sigma=2, radius=2)
            aff = Image.fromarray((remove_noise2 * 255).astype(np.uint8))  # Scale to 0-255
            canny_edge = cv2.Canny(np.array(aff), 20, 70)
            self.show_image(canny_edge)  # hiển thị ảnh

################################################################################################################################
# Khởi chạy giao diện
root = tk.Tk()
app = ImageProcessingApp(root)
root.mainloop()
