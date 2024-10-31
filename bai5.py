# bai5.py
import pandas as pd
from decision_tree import build_cart_tree, build_id3_tree, predict, get_split

# Hàm đọc dữ liệu từ tệp CSV
def load_data(file_path):
    dataset = pd.read_csv(file_path)
    return dataset.values.tolist()

# Hàm chạy CART
def run_cart(dataset):
    max_depth = 5
    min_size = 1
    tree = build_cart_tree(get_split(dataset), max_depth, min_size)  # Sử dụng get_split ở đây
    print("CART Tree: ", tree)
    return tree

# Hàm chạy ID3
def run_id3(dataset):
    attributes = list(range(len(dataset[0]) - 1))  # Chọn các chỉ số thuộc tính
    tree = build_id3_tree(dataset, attributes)
    print("ID3 Tree: ", tree)
    return tree

# Hàm dự đoán từ đầu vào của người dùng
def get_user_input():
    try:
        sepal_length = float(input("Nhập chiều dài đài (Sepal Length) cm: "))
        sepal_width = float(input("Nhập chiều rộng đài (Sepal Width) cm: "))
        petal_length = float(input("Nhập chiều dài cánh (Petal Length) cm: "))
        petal_width = float(input("Nhập chiều rộng cánh (Petal Width) cm: "))
        return [sepal_length, sepal_width, petal_length, petal_width]
    except ValueError:
        print("Giá trị không hợp lệ, vui lòng nhập lại.")
        return get_user_input()

# Hàm dự đoán loại hoa
def make_prediction(tree, user_input):
    prediction = predict(tree, user_input + [''])  # Thêm một giá trị rỗng cho chỉ số class
    return prediction

if __name__ == "__main__":
    # Tải dữ liệu IRIS với ký tự escape trong đường dẫn được xử lý
    iris_data = load_data(r'D:\XLA\day_31\B5\iris.csv')
    
    print("Running CART...")
    cart_tree = run_cart(iris_data)

    print("\nRunning ID3...")
    id3_tree = run_id3(iris_data)

    # Nhập dữ liệu từ người dùng
    user_input = get_user_input()

    # Dự đoán với cây CART
    cart_prediction = make_prediction(cart_tree, user_input)
    print(f"Dự đoán với CART: {cart_prediction}")

    # Dự đoán với cây ID3
    id3_prediction = make_prediction(id3_tree, user_input)
    print(f"Dự đoán với ID3: {id3_prediction}")
