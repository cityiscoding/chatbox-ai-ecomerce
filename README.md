#E-Commerce Chatbot sử dụng NLTK D#ưới đây là chi tiết về quy trình thực hiện:

Tiền xử lý văn bản với NLTK

Chuyển toàn bộ văn bản thành chữ hoa hoặc chữ thường, để thuật toán không xử lý các từ giống nhau ở các trường hợp khác nhau như là khác nhau
Tokenization: Tokenization chỉ là thuật ngữ được sử dụng để mô tả quá trình chuyển đổi các chuỗi văn bản thông thường thành một danh sách các token, tức là các từ chúng ta thực sự muốn. Tokenizer câu có thể được sử dụng để tìm danh sách các câu và Tokenizer từ có thể được sử dụng để tìm danh sách các từ trong chuỗi.
Bộ dữ liệu NLTK bao gồm một Punkt tokenizer đã được huấn luyện trước cho tiếng việt.
[+] Phương pháp TF-IDF: TF : Tần suất từ (Số lần từ xuất hiện trong một tài liệu) IDF : Ngược tần suất tài liệu (Tần suất từ hiếm qua các tài liệu)

TF = (Số lần từ t xuất hiện trong một tài liệu)/(Số lượng từ trong tài liệu) IDF = 1+log(N/n), trong đó N là số tài liệu và n là số tài liệu mà từ t đã xuất hiện. [+] Tương đồng Cosine: Tương đồng cosine là một độ đo tương đồng giữa hai vector khác không. Sử dụng công thức này, chúng ta có thể xác định sự tương đồng giữa hai tài liệu bất kỳ d1 và d2. Tương đồng Cosine (d1, d2) = Tích vô hướng(d1, d2) / ||d1|| * ||d2|| trong đó d1 và d2 là hai vector khác không

TẬP TIN

[+] Intents.json – Tệp dữ liệu chứa các mẫu và phản hồi đã được định nghĩa trước. [+] train_chatbot.py – Trong tệp Python này, chúng tôi đã viết một kịch bản để xây dựng mô hình và đào tạo chatbot của chúng tôi. [+] Words.pkl – Đây là một tệp pickle trong đó chúng tôi lưu trữ đối tượng Python chứa một danh sách từ vựng của chúng tôi. [+] Classes.pkl – Tệp pickle classes chứa danh sách các danh mục. [+] Chatbot_model.h5 – Đây là mô hình đã được đào tạo chứa thông tin về mô hình và có trọng số của các nơ-ron.

#Cấu trúc của mô hình

[+] Nhập và tải tệp dữ liệu [+] Tiền xử lý dữ liệu [+] Tạo dữ liệu đào tạo và kiểm thử [+] Xây dựng mô hình [+] Dự đoán phản hồi

#Hướng dẫn chạy chương trình chatbox yêu cầu phiên bản python tối thiểu là 3.9 để có hiệu suất tốt và dể dàng cài đặt thư viện Bước 1: Dùng dòng lệnh : pip install -r requirements.txt Để cài đặt thư viện cần thiết cho chương trình chatbox Bước 2: Dùng dòng lệnh: python chatbox.py Để chạy chương trình. #Giải thích bước 2 Code chương trình đã được tinh chỉnh lại chỉ cần 1 lần chạy. Chương trình sẽ tự xữ lý theo trình tự cấu trúc của mô hình.
