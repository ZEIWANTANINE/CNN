import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
import psycopg2
import logging
from typing import List, Dict
import unicodedata
from datetime import date, datetime, timedelta
import re
import subprocess
from fastapi import Query

#python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Cấu hình CORS
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thông tin kết nối PostgreSQL
DB_CONFIG_CRAWLER = {
    "host": "localhost",
    "port": "5432",
    "database": "crawler_posts",
    "user": "postgres",
    "password": "0311"
}

DB_CONFIG_CLASSIFY = {
    "host": "localhost",
    "port": "5432",
    "database": "classify_posts",
    "user": "postgres",
    "password": "0311"
}

# Model để nhận dữ liệu từ frontend
class InputText(BaseModel):
    text: str

class CrawlRequest(BaseModel):
    num_posts: int
    fanpage_link: str

# Định nghĩa mô hình ConvNet
class ConvNet(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.2):
        super(ConvNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        logits = self.fc3(x)
        probs = torch.softmax(logits, dim=1)
        return logits, probs

# Tải PhoBERT và mô hình ConvNet
try:
    logger.info("Đang tải PhoBERT để trích xuất đặc trưng...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    phobert_model = AutoModel.from_pretrained("vinai/phobert-base").to(device)
    phobert_model.eval()

    logger.info("Đang tải mô hình ConvNet từ file .pth...")
    input_size = 768
    output_size = 3
    model = ConvNet(input_size=input_size, output_size=output_size).to(device)

    model_path = "trained_model.pth"
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Đã tải PhoBERT và mô hình ConvNet thành công")
except Exception as e:
    logger.error(f"Lỗi khi tải PhoBERT hoặc mô hình: {e}")
    raise Exception("Không thể tải mô hình")

# Hàm chuẩn hóa văn bản
def normalize_text(text: str, max_char_length=1500, max_word_count=400) -> str:
    try:
        if len(text) > max_char_length:
            text = text[:max_char_length]
            logger.warning(f"Văn bản quá dài, đã cắt xuống {max_char_length} ký tự: {text[:50]}...")
        words = text.split()
        if len(words) > max_word_count:
            text = ' '.join(words[:max_word_count])
            logger.warning(f"Văn bản có quá nhiều từ, đã cắt xuống {max_word_count} từ: {text[:50]}...")
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        text = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f]', '', text)
        text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]', '', text)
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'[^\w\s.,!?]', '', text)
        text = ' '.join(text.split())
        return text
    except Exception as e:
        logger.error(f"Lỗi khi chuẩn hóa văn bản: {e}")
        return ""

# Hàm kiểm tra số lượng token
def check_token_length(text: str, tokenizer, max_length: int) -> bool:
    try:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return len(tokens) <= max_length
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra số lượng token: {e}")
        return False

# Hàm trích xuất đặc trưng
def extract_text_features(texts, device, phobert_model, tokenizer, batch_size=16, max_length=512):
    all_features = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_texts = [text if text.strip() != "" else "[EMPTY]" for text in batch_texts]
        for text in batch_texts:
            if not check_token_length(text, tokenizer, max_length):
                logger.warning(f"Văn bản quá dài sau token hóa: {text[:50]}... Sử dụng giá trị mặc định.")
                return np.zeros((len(batch_texts), 768))
        try:
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=max_length).to(device)
            with torch.no_grad():
                model_output = phobert_model(**encoded_input)
            batch_features = model_output.last_hidden_state[:, 0, :].cpu().numpy()
            all_features.append(batch_features)
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất đặc trưng: {e}")
            return np.zeros((len(batch_texts), 768))
        torch.cuda.empty_cache()
    return np.concatenate(all_features, axis=0)

# Hàm kết nối đến PostgreSQL
def connect_to_db(db_config):
    try:
        conn = psycopg2.connect(**db_config)
        logger.info(f"Kết nối đến PostgreSQL ({db_config['database']}) thành công")
        return conn
    except Exception as e:
        logger.error(f"Lỗi khi kết nối đến PostgreSQL ({db_config['database']}): {e}")
        return None

# Hàm kiểm tra URL trùng lặp trong classified_posts
def check_url_exists_in_classified(conn, url):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM classified_posts WHERE url = %s", (url,))
            exists = cur.fetchone() is not None
        return exists
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra URL trong bảng classified_posts: {e}")
        return False

# Hàm kiểm tra URL trùng lặp trong crawler_posts
def check_url_exists_in_crawler(conn, url):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM crawler_posts WHERE url = %s", (url,))
            exists = cur.fetchone() is not None
        return exists
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra URL trong bảng crawler_posts: {e}")
        return False

# Hàm kiểm tra text trùng lặp trong classified_posts
def check_text_exists_in_classified(conn, text):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM classified_posts WHERE text = %s", (text,))
            exists = cur.fetchone() is not None
        return exists
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra text trong bảng classified_posts: {e}")
        return False

# Hàm lưu dữ liệu thô vào crawler_posts
def save_to_crawler_posts(conn, url: str, text: str):
    try:
        if url != "Manually entered" and check_url_exists_in_crawler(conn, url):
            logger.info(f"URL đã tồn tại trong crawler_posts: {url}. Bỏ qua bài viết này.")
            return
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO crawler_posts (url, text) VALUES (%s, %s) RETURNING id",
                (url, text)
            )
            row = cur.fetchone()
            if row:
                logger.info(f"Đã lưu bài viết vào crawler_posts: {text[:50]}... (URL: {url}, ID: {row[0]})")
            else:
                logger.warning(f"Không lưu được dữ liệu vào crawler_posts: {url}")
        conn.commit()
    except Exception as e:
        logger.error(f"Lỗi khi lưu vào crawler_posts: {e}")
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu vào crawler_posts: {str(e)}")

# Hàm lưu bài viết đã phân loại vào classified_posts
def save_classified_text(conn_classify, text: str, label: str, url: str = "Manually entered"):
    try:
        neutral = label == "neutral"
        negative = label == "negative"
        positive = label == "positive"

        if url != "Manually entered" and check_text_exists_in_classified(conn_classify, text):
            logger.info(f"Văn bản đã tồn tại trong classified_posts: {text[:50]}... (URL: {url}). Bỏ qua bài viết này.")
            return

        with conn_classify.cursor() as cur:
            cur.execute(
                """
                INSERT INTO classified_posts (url, text, neutral, negative, positive)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (url, text, neutral, negative, positive)
            )
            row = cur.fetchone()
            if row:
                logger.info(f"Đã lưu vào classified_posts: {text[:50]}... (Label: {label}, URL: {url}, ID: {row[0]})")
            else:
                logger.warning(f"Không lưu được dữ liệu vào classified_posts: {url}")
        conn_classify.commit()
    except Exception as e:
        logger.error(f"Lỗi khi lưu vào classified_posts: {e}")
        conn_classify.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu vào classified_posts: {str(e)}")

# Hàm chạy mô hình AI để phân loại
def run_model_inference(input_text: str) -> tuple[str, str]:
    try:
        normalized_text = normalize_text(input_text, max_char_length=1500, max_word_count=400)
        logger.info(f"Văn bản sau khi chuẩn hóa: {normalized_text[:50]}...")
        if not normalized_text.strip():
            logger.warning("Văn bản rỗng sau khi chuẩn hóa")
            return normalized_text, "neutral"
        features = extract_text_features([normalized_text], device, phobert_model, tokenizer, batch_size=1, max_length=512)
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits, probs = model(features_tensor)
            prediction = torch.argmax(logits, dim=1).item()
        label_map = {0: "neutral", 1: "positive", 2: "negative"}
        label = label_map[prediction]
        logger.info(f"Kết quả phân loại: {label}")
        return normalized_text, label
    except Exception as e:
        logger.error(f"Lỗi khi chạy mô hình AI: {str(e)}")
        return input_text, "neutral"

# Hàm phân loại các bài viết mới từ crawler_posts và lưu vào classified_posts
def classify_new_posts():
    try:
        conn_crawler = connect_to_db(DB_CONFIG_CRAWLER)
        if not conn_crawler:
            raise HTTPException(status_code=500, detail="Không thể kết nối đến PostgreSQL (crawler_posts)")

        conn_classify = connect_to_db(DB_CONFIG_CLASSIFY)
        if not conn_classify:
            raise HTTPException(status_code=500, detail="Không thể kết nối đến PostgreSQL (classify_posts)")

        # Bước 1: Lấy MAX(created_at) từ classified_posts
        max_created_at = '1970-01-01'  # Giá trị mặc định
        with conn_classify.cursor() as cur:
            cur.execute("SELECT COALESCE(MAX(created_at), '1970-01-01') FROM classified_posts")
            result = cur.fetchone()
            if result:
                max_created_at = result[0]

        # Bước 2: Lấy các bài viết mới từ crawler_posts
        with conn_crawler.cursor() as cur:
            cur.execute("SELECT url, text FROM crawler_posts WHERE created_at > %s", (max_created_at,))
            new_posts = cur.fetchall()
        if not new_posts:
            logger.info("Không có bài viết mới trong crawler_posts để phân loại")
            return 0

        classified_count = 0
        for url, text in new_posts:
            if url != "Manually entered" and check_url_exists_in_classified(conn_classify, url):
                logger.info(f"Phát hiện URL trùng lặp trong classified_posts: {url}. Bỏ qua bài viết này.")
                continue
            
            truncated_text, label = run_model_inference(text)
            if url != "Manually entered" and check_text_exists_in_classified(conn_classify, truncated_text):
                logger.info(f"Văn bản đã tồn tại trong classified_posts: {truncated_text[:50]}... (URL: {url})")
                continue
            
            save_classified_text(conn_classify, truncated_text, label, url)
            classified_count += 1
            logger.info(f"Đã phân loại bài viết: {truncated_text[:50]}... (Label: {label}, URL: {url})")

        conn_crawler.close()
        conn_classify.close()
        return classified_count
    except Exception as e:
        logger.error(f"Lỗi khi phân loại bài viết mới: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi phân loại bài viết mới: {str(e)}")

def get_all_urls_from_crawler(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT url FROM crawler_posts")
            urls = {row[0] for row in cur.fetchall()}
        logger.info(f"Đã lấy {len(urls)} URL từ crawler_posts")
        return urls
    except Exception as e:
        logger.error(f"Lỗi khi lấy URL từ crawler_posts: {str(e)}")
        raise

def get_all_urls_from_classified(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT url FROM classified_posts")
            urls = {row[0] for row in cur.fetchall()}
        logger.info(f"Đã lấy {len(urls)} URL từ classified_posts")
        return urls
    except Exception as e:
        logger.error(f"Lỗi khi lấy URL từ classified_posts: {str(e)}")
        raise

# Hàm crawl và phân loại
def run_crawler(num_posts: int, fanpage_link: str) -> List[Dict]:
    try:
        logger.info(f"Đang chạy crawler để lấy {num_posts} bài viết từ fanpage: {fanpage_link}...")
        result = subprocess.run(
            ["python3", "crawler.py", str(num_posts), fanpage_link],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("Crawler đã chạy xong")

        # Parse danh sách bài viết từ stdout của crawler.py
        crawled_posts = json.loads(result.stdout)

        # Kết nối database
        conn_crawler = connect_to_db(DB_CONFIG_CRAWLER)
        conn_classify = connect_to_db(DB_CONFIG_CLASSIFY)
        if not conn_crawler or not conn_classify:
            raise HTTPException(status_code=500, detail="Không thể kết nối đến PostgreSQL")

        # Lấy tất cả URL từ cả hai bảng một lần duy nhất
        existing_crawler_urls = get_all_urls_from_crawler(conn_crawler)
        existing_classified_urls = get_all_urls_from_classified(conn_classify)

        classified_posts = []
        for post in crawled_posts:
            url = post["url"]
            sub_posts = post["sub_posts"]

            # Kiểm tra trùng lặp URL trên bài viết gốc
            is_duplicate_in_crawler = url in existing_crawler_urls
            is_duplicate_in_classified = url in existing_classified_urls
            is_duplicate = is_duplicate_in_crawler or is_duplicate_in_classified

            # Nếu bài viết gốc không trùng, lưu tất cả bài con
            if not is_duplicate:
                for sub_post in sub_posts:
                    text = sub_post["text"]
                    save_to_crawler_posts(conn_crawler, url, text)
                existing_crawler_urls.add(url)  # Cập nhật danh sách trong bộ nhớ

            # Phân loại và lưu tất cả bài con nếu không trùng
            for sub_post in sub_posts:
                text = sub_post["text"]
                truncated_text, label = run_model_inference(text)

                if not is_duplicate:
                    save_classified_text(conn_classify, truncated_text, label, url)
                    existing_classified_urls.add(url)  # Cập nhật danh sách trong bộ nhớ

                classified_posts.append({
                    "url": url,
                    "text": truncated_text,
                    "label": label,
                    "is_duplicate": is_duplicate
                })

        conn_crawler.close()
        conn_classify.close()
        logger.info(f"Đã xử lý {len(classified_posts)} bài viết crawl được")
        return classified_posts

    except subprocess.CalledProcessError as e:
        error_message = "Lỗi không xác định khi chạy crawler"
        if e.stderr:
            error_lines = e.stderr.splitlines()
            for line in error_lines:
                if "ERROR" in line and "Lỗi khi chạy actor" in line:
                    error_message = line.split("Lỗi khi chạy actor: ", 1)[-1].strip()
                    break
                elif "ERROR" in line and "Lỗi trong crawl_and_save" in line:
                    error_message = line.split("Lỗi trong crawl_and_save: ", 1)[-1].strip()
                    break
        logger.error(f"Lỗi khi chạy crawler: {error_message}")
        raise HTTPException(status_code=500, detail=error_message)

    except Exception as e:
        logger.error(f"Lỗi khi chạy crawler: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Hàm lấy tất cả dữ liệu từ crawler_posts với phân trang
def get_all_crawler_posts(page: int = 1, limit: int = 10):
    try:
        conn_crawler = connect_to_db(DB_CONFIG_CRAWLER)
        if not conn_crawler:
            raise HTTPException(status_code=500, detail="Không thể kết nối đến PostgreSQL (crawler_posts)")

        # Tính offset
        offset = (page - 1) * limit

        # Lấy tổng số bài viết
        with conn_crawler.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM crawler_posts")
            total = cur.fetchone()[0]

        # Lấy dữ liệu phân trang
        with conn_crawler.cursor() as cur:
            cur.execute(
                "SELECT id, url, text, created_at FROM crawler_posts ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (limit, offset)
            )
            rows = cur.fetchall()
            posts = [
                {
                    "id": row[0],
                    "url": row[1] if row[1] else "",
                    "text": row[2],
                    "created_at": row[3].isoformat() if row[3] else None
                } for row in rows
            ]

        conn_crawler.close()
        total_pages = (total + limit - 1) // limit  # Tính tổng số trang
        logger.info(f"Đã lấy {len(posts)} bài viết từ crawler_posts (trang {page}, limit {limit})")
        return {
            "posts": posts,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": total_pages
        }
    except Exception as e:
        logger.error(f"Lỗi khi lấy dữ liệu từ crawler_posts: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy dữ liệu từ crawler_posts: {str(e)}")

# Hàm lấy các bài viết tích cực từ classified_posts với phân trang
def get_positive_posts(page: int = 1, limit: int = 10):
    try:
        conn_classify = connect_to_db(DB_CONFIG_CLASSIFY)
        if not conn_classify:
            raise HTTPException(status_code=500, detail="Không thể kết nối đến PostgreSQL (classify_posts)")

        # Tính offset
        offset = (page - 1) * limit

        # Lấy tổng số bài viết tích cực
        with conn_classify.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM classified_posts WHERE positive = true")
            total = cur.fetchone()[0]

        # Lấy dữ liệu phân trang
        with conn_classify.cursor() as cur:
            cur.execute(
                """
                SELECT id, url, text, neutral, negative, positive, created_at 
                FROM classified_posts 
                WHERE positive = true 
                ORDER BY created_at DESC 
                LIMIT %s OFFSET %s
                """,
                (limit, offset)
            )
            rows = cur.fetchall()
            posts = [
                {
                    "id": row[0],
                    "url": row[1] if row[1] else "",
                    "text": row[2],
                    "neutral": row[3],
                    "negative": row[4],
                    "positive": row[5],
                    "created_at": row[6].isoformat() if row[6] else None
                } for row in rows
            ]

        conn_classify.close()
        total_pages = (total + limit - 1) // limit  # Tính tổng số trang
        logger.info(f"Đã lấy {len(posts)} bài viết tích cực từ classified_posts (trang {page}, limit {limit})")
        return {
            "posts": posts,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": total_pages
        }
    except Exception as e:
        logger.error(f"Lỗi khi lấy bài viết tích cực từ classified_posts: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy bài viết tích cực từ classified_posts: {str(e)}")

# Hàm lấy các bài viết tiêu cực từ classified_posts với phân trang
def get_negative_posts(page: int = 1, limit: int = 10):
    try:
        conn_classify = connect_to_db(DB_CONFIG_CLASSIFY)
        if not conn_classify:
            raise HTTPException(status_code=500, detail="Không thể kết nối đến PostgreSQL (classify_posts)")

        # Tính offset
        offset = (page - 1) * limit

        # Lấy tổng số bài viết tiêu cực
        with conn_classify.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM classified_posts WHERE negative = true")
            total = cur.fetchone()[0]

        # Lấy dữ liệu phân trang
        with conn_classify.cursor() as cur:
            cur.execute(
                """
                SELECT id, url, text, neutral, negative, positive, created_at 
                FROM classified_posts 
                WHERE negative = true 
                ORDER BY created_at DESC 
                LIMIT %s OFFSET %s
                """,
                (limit, offset)
            )
            rows = cur.fetchall()
            posts = [
                {
                    "id": row[0],
                    "url": row[1] if row[1] else "",
                    "text": row[2],
                    "neutral": row[3],
                    "negative": row[4],
                    "positive": row[5],
                    "created_at": row[6].isoformat() if row[6] else None
                } for row in rows
            ]

        conn_classify.close()
        total_pages = (total + limit - 1) // limit  # Tính tổng số trang
        logger.info(f"Đã lấy {len(posts)} bài viết tiêu cực từ classified_posts (trang {page}, limit {limit})")
        return {
            "posts": posts,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": total_pages
        }
    except Exception as e:
        logger.error(f"Lỗi khi lấy bài viết tiêu cực từ classified_posts: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy bài viết tiêu cực từ classified_posts: {str(e)}")

# Hàm lấy các bài viết trung lập từ classified_posts với phân trang
def get_neutral_posts(page: int = 1, limit: int = 10):
    try:
        conn_classify = connect_to_db(DB_CONFIG_CLASSIFY)
        if not conn_classify:
            raise HTTPException(status_code=500, detail="Không thể kết nối đến PostgreSQL (classify_posts)")

        # Tính offset
        offset = (page - 1) * limit

        # Lấy tổng số bài viết trung lập
        with conn_classify.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM classified_posts WHERE neutral = true")
            total = cur.fetchone()[0]

        # Lấy dữ liệu phân trang
        with conn_classify.cursor() as cur:
            cur.execute(
                """
                SELECT id, url, text, neutral, negative, positive, created_at 
                FROM classified_posts 
                WHERE neutral = true 
                ORDER BY created_at DESC 
                LIMIT %s OFFSET %s
                """,
                (limit, offset)
            )
            rows = cur.fetchall()
            posts = [
                {
                    "id": row[0],
                    "url": row[1] if row[1] else "",
                    "text": row[2],
                    "neutral": row[3],
                    "negative": row[4],
                    "positive": row[5],
                    "created_at": row[6].isoformat() if row[6] else None
                } for row in rows
            ]

        conn_classify.close()
        total_pages = (total + limit - 1) // limit  # Tính tổng số trang
        logger.info(f"Đã lấy {len(posts)} bài viết trung lập từ classified_posts (trang {page}, limit {limit})")
        return {
            "posts": posts,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": total_pages
        }
    except Exception as e:
        logger.error(f"Lỗi khi lấy bài viết trung lập từ classified_posts: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy bài viết trung lập từ classified_posts: {str(e)}")

# API để phân loại nội dung (manual entry)
@app.post("/classify")
async def classify(data: InputText):
    conn_crawler = connect_to_db(DB_CONFIG_CRAWLER)
    conn_classify = connect_to_db(DB_CONFIG_CLASSIFY)
    if not conn_crawler or not conn_classify:
        raise HTTPException(status_code=500, detail="Không thể kết nối đến PostgreSQL")

    try:
        url = "Manually entered"
        truncated_text, result = run_model_inference(data.text)

        if check_text_exists_in_classified(conn_classify, truncated_text):
            logger.info(f"Văn bản đã tồn tại trong classified_posts: {truncated_text[:50]}... (URL: {url}). Không lưu vào database.")
            return {"classification": result, "is_duplicate": True}

        save_to_crawler_posts(conn_crawler, url, data.text)
        save_classified_text(conn_classify, truncated_text, result, url)
        logger.info(f"Đã lưu văn bản mới vào database: {truncated_text[:50]}... (Label: {result})")
        return {"classification": result, "is_duplicate": False}
    finally:
        conn_crawler.close()
        conn_classify.close()

# API để chạy crawler
@app.post("/crawl")
async def crawl(request: CrawlRequest):
    num_posts = request.num_posts
    fanpage_link = request.fanpage_link
    if num_posts <= 0:
        raise HTTPException(status_code=400, detail="Số lượng bài viết phải lớn hơn 0")
    if not fanpage_link:
        raise HTTPException(status_code=400, detail="Link fanpage không được để trống")
    posts = run_crawler(num_posts, fanpage_link)
    return {"posts": posts}

# API để lấy tất cả bài viết đã phân loại
@app.get("/posts")
async def get_posts(page: int = Query(1, ge=1), limit: int = Query(10, ge=1)):
    try:
        conn_classify = connect_to_db(DB_CONFIG_CLASSIFY)
        if not conn_classify:
            raise HTTPException(status_code=500, detail="Không thể kết nối đến PostgreSQL (classify_posts)")

        offset = (page - 1) * limit

        with conn_classify.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM classified_posts")
            total = cur.fetchone()[0]

        with conn_classify.cursor() as cur:
            cur.execute(
                """
                SELECT id, url, text, neutral, negative, positive, created_at 
                FROM classified_posts 
                ORDER BY created_at DESC 
                LIMIT %s OFFSET %s
                """,
                (limit, offset)
            )
            rows = cur.fetchall()
            posts = [
                {
                    "id": row[0],
                    "url": row[1] if row[1] else "",
                    "text": row[2],
                    "neutral": row[3],
                    "negative": row[4],
                    "positive": row[5],
                    "created_at": row[6].isoformat() if row[6] else None
                } for row in rows
            ]

        conn_classify.close()
        total_pages = (total + limit - 1) // limit
        logger.info(f"Đã lấy {len(posts)} bài viết từ classified_posts (trang {page}, limit {limit})")
        return {
            "posts": posts,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": total_pages
        }
    except Exception as e:
        logger.error(f"Lỗi khi lấy dữ liệu từ classified_posts: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy dữ liệu từ classified_posts: {str(e)}")

# API để lấy tất cả dữ liệu từ crawler_posts
@app.get("/crawler-posts")
async def get_crawler_posts(page: int = Query(1, ge=1), limit: int = Query(10, ge=1)):
    return get_all_crawler_posts(page, limit)

# API để lấy các bài viết tích cực từ classified_posts
@app.get("/positive-posts")
async def get_positive_classified_posts(page: int = Query(1, ge=1), limit: int = Query(10, ge=1)):
    return get_positive_posts(page, limit)

# API để lấy các bài viết tiêu cực từ classified_posts
@app.get("/negative-posts")
async def get_negative_classified_posts(page: int = Query(1, ge=1), limit: int = Query(10, ge=1)):
    return get_negative_posts(page, limit)

# API để lấy các bài viết trung lập từ classified_posts
@app.get("/neutral-posts")
async def get_neutral_classified_posts(page: int = Query(1, ge=1), limit: int = Query(10, ge=1)):
    return get_neutral_posts(page, limit)

# API sửa bài viết trong crawler_posts
@app.put("/crawler-posts/{id}")
async def update_crawler_post(id: int, data: InputText):
    try:
        conn_crawler = connect_to_db(DB_CONFIG_CRAWLER)
        if not conn_crawler:
            raise HTTPException(status_code=500, detail="Không thể kết nối đến PostgreSQL (crawler_posts)")

        with conn_crawler.cursor() as cur:
            cur.execute(
                "UPDATE crawler_posts SET text = %s WHERE id = %s RETURNING id",
                (data.text, id)
            )
            updated_crawler = cur.fetchone()
            if not updated_crawler:
                raise HTTPException(status_code=404, detail=f"Không tìm thấy bài viết với ID {id} trong crawler_posts")
        conn_crawler.commit()

        logger.info(f"Đã cập nhật bài viết ID {id} trong crawler_posts: {data.text[:50]}...")
        return {"message": "Cập nhật bài viết thành công trong crawler_posts"}
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật bài viết ID {id} trong crawler_posts: {e}")
        conn_crawler.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi khi cập nhật bài viết: {str(e)}")
    finally:
        conn_crawler.close()

# API sửa bài viết trong classified_posts
@app.put("/classified-posts/{id}")
async def update_classified_post(id: int, data: InputText):
    try:
        conn_classify = connect_to_db(DB_CONFIG_CLASSIFY)
        if not conn_classify:
            raise HTTPException(status_code=500, detail="Không thể kết nối đến PostgreSQL (classify_posts)")

        truncated_text, label = run_model_inference(data.text)
        if not truncated_text.strip():
            raise HTTPException(status_code=400, detail="Văn bản không hợp lệ sau khi chuẩn hóa")

        neutral = label == "neutral"
        negative = label == "negative"
        positive = label == "positive"
        with conn_classify.cursor() as cur:
            cur.execute(
                """
                UPDATE classified_posts 
                SET text = %s, neutral = %s, negative = %s, positive = %s 
                WHERE id = %s RETURNING id
                """,
                (truncated_text, neutral, negative, positive, id)
            )
            updated_classified = cur.fetchone()
            if not updated_classified:
                raise HTTPException(status_code=404, detail=f"Không tìm thấy bài viết với ID {id} trong classified_posts")
        conn_classify.commit()

        logger.info(f"Đã cập nhật bài viết ID {id} trong classified_posts: {truncated_text[:50]}... (Label: {label})")
        return {"message": "Cập nhật bài viết thành công trong classified_posts", "classification": label}
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật bài viết ID {id} trong classified_posts: {e}")
        conn_classify.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi khi cập nhật bài viết: {str(e)}")
    finally:
        conn_classify.close()

# API xoá bài viết trong classified_posts
@app.delete("/posts/{id}")
async def delete_post(id: int):
    try:
        conn_crawler = connect_to_db(DB_CONFIG_CRAWLER)
        conn_classify = connect_to_db(DB_CONFIG_CLASSIFY)
        if not conn_crawler or not conn_classify:
            raise HTTPException(status_code=500, detail="Không thể kết nối đến PostgreSQL")

        with conn_crawler.cursor() as cur:
            cur.execute("DELETE FROM crawler_posts WHERE id = %s RETURNING id", (id,))
            deleted_crawler = cur.fetchone()
            if not deleted_crawler:
                logger.warning(f"Không tìm thấy bài viết với ID {id} trong crawler_posts")
        conn_crawler.commit()

        with conn_classify.cursor() as cur:
            cur.execute("DELETE FROM classified_posts WHERE id = %s RETURNING id", (id,))
            deleted_classified = cur.fetchone()
            if not deleted_classified:
                logger.warning(f"Không tìm thấy bài viết với ID {id} trong classified_posts")
        conn_classify.commit()

        if not deleted_crawler and not deleted_classified:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy bài viết với ID {id}")

        logger.info(f"Đã xóa bài viết ID {id}")
        return {"message": "Xóa bài viết thành công"}
    except Exception as e:
        logger.error(f"Lỗi khi xóa bài viết ID {id}: {e}")
        conn_crawler.rollback()
        conn_classify.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa bài viết: {str(e)}")
    finally:
        conn_crawler.close()
        conn_classify.close()

# API thống kê bài viết theo nhãn trong classified_posts
@app.get("/stats/classified-posts/labels")
async def get_label_stats():
    try:
        conn_classify = connect_to_db(DB_CONFIG_CLASSIFY)
        if not conn_classify:
            raise HTTPException(status_code=500, detail="Không thể kết nối đến PostgreSQL")

        with conn_classify.cursor() as cur:
            cur.execute("""
                SELECT 
                    SUM(CASE WHEN neutral THEN 1 ELSE 0 END) as neutral_count,
                    SUM(CASE WHEN negative THEN 1 ELSE 0 END) as negative_count,
                    SUM(CASE WHEN positive THEN 1 ELSE 0 END) as positive_count
                FROM classified_posts
            """)
            result = cur.fetchone()
            stats = {
                "neutral": result[0] or 0,
                "negative": result[1] or 0,
                "positive": result[2] or 0
            }
        conn_classify.commit()
        return stats
    except Exception as e:
        logger.error(f"Lỗi khi lấy thống kê nhãn: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy thống kê: {str(e)}")
    finally:
        conn_classify.close()

# API thống kê bài viết theo thời gian trong classified_posts
@app.get("/stats/classified-posts/by-date")
async def get_posts_by_date(label: str = None, start_date: str = None, end_date: str = None):
    try:
        conn_classify = connect_to_db(DB_CONFIG_CLASSIFY)
        if not conn_classify:
            raise HTTPException(status_code=500, detail="Không thể kết nối đến PostgreSQL")

        if not start_date:
            with conn_classify.cursor() as cur:
                cur.execute("SELECT MIN(created_at) FROM classified_posts")
                min_date = cur.fetchone()[0]
                if min_date:
                    start_date = min_date.strftime("%Y-%m-%d")
                else:
                    start_date = date.today().strftime("%Y-%m-%d")
                
        if not end_date:
            end_date = date.today().strftime("%Y-%m-%d")

        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        end_date_next_day = (end_date_obj + timedelta(days=1)).strftime("%Y-%m-%d")

        query = """
            SELECT DATE(created_at) as date, 
                   COUNT(*) as count
            FROM classified_posts
            WHERE 1=1
        """
        params = []
        if label:
            query += f" AND {label} = TRUE"
        if start_date:
            query += " AND created_at >= %s"
            params.append(start_date)
        if end_date:
            query += " AND created_at < %s"
            params.append(end_date_next_day)
        query += " GROUP BY DATE(created_at) ORDER BY DATE(created_at)"

        with conn_classify.cursor() as cur:
            cur.execute(query, params)
            results = cur.fetchall()
            stats = [{"date": str(row[0]), "count": row[1]} for row in results]
        conn_classify.commit()
        return stats
    except Exception as e:
        logger.error(f"Lỗi khi lấy thống kê theo ngày: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy thống kê: {str(e)}")
    finally:
        if conn_classify:
            conn_classify.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)