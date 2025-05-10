### Yêu cầu hệ thống
- Python 3.9+
- PostgreSQL database
- Google Gemini API key

### Thiết lập biến môi trường:
Tạo file `.env` trong thư mục gốc của dự án và thêm các biến môi trường sau:
```
DATABASE_URL=postgresql://username:password@host:port/dbname
GEMINI_API_KEY=your-gemini-api-key
```

### Tải các packages
pip install -r requirements.txt

### Chạy ứng dụng
python main.py
