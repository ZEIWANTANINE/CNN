import sys
import logging
import psycopg2
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import pickle
from bs4 import BeautifulSoup
import re
import json

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cấu hình database
DB_CONFIG_CRAWLER = {
    "host": "localhost",
    "port": "5432",
    "database": "crawler_posts",
    "user": "postgres",
    "password": "0311"
}

# Thông tin đăng nhập Facebook
FACEBOOK_EMAIL = "email" # thay bằng tài khoản fb 
FACEBOOK_PASSWORD = "password"

# Hàm giả lập hành vi nhập liệu giống con người
def human_like_typing(element, text):
    for char in text:
        element.send_keys(char)
        time.sleep(random.uniform(0.05, 0.2))

# Hàm kết nối database
def connect_to_db():
    try:
        conn = psycopg2.connect(**DB_CONFIG_CRAWLER)
        logger.info("Kết nối đến PostgreSQL (crawler_posts) thành công")
        return conn
    except Exception as e:
        logger.error(f"Lỗi khi kết nối đến PostgreSQL (crawler_posts): {str(e)}")
        raise

# Hàm kiểm tra URL tồn tại
def check_url_exists(conn, url):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM crawler_posts WHERE url = %s", (url,))
            exists = cur.fetchone() is not None
        return exists
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra URL: {str(e)}")
        return False

# Hàm lưu bài viết vào database
def save_to_posts(conn, url: str, text: str):
    try:
        if check_url_exists(conn, url):
            logger.info(f"URL đã tồn tại: {url}. Bỏ qua.")
            return False
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO crawler_posts (url, text) VALUES (%s, %s) RETURNING id",
                (url, text)
            )
            row = cur.fetchone()
            if row:
                logger.info(f"Đã lưu bài viết: {text[:50]}... (URL: {url}, ID: {row[0]})")
            else:
                logger.warning(f"Không lưu được dữ liệu: {url}")
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Lỗi khi lưu bài viết: {str(e)}")
        conn.rollback()
        raise

# Hàm nhấn nút "Xem thêm" trong bài viết
def click_see_more_in_post(driver, post_element):
    try:
        see_more_buttons = post_element.find_elements(By.XPATH, ".//div[contains(@class, 'x1i10hfl') and (contains(text(), 'Xem thêm') or contains(text(), 'See more'))]")
        if not see_more_buttons:
            logger.info("Không tìm thấy nút 'Xem thêm' trong bài viết.")
            return False

        for button in see_more_buttons:
            try:
                driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", button)
                time.sleep(random.uniform(0.5, 1))
                is_visible = driver.execute_script(
                    "var elem = arguments[0], box = elem.getBoundingClientRect(), "
                    "cx = box.left + box.width / 2, cy = box.top + box.height / 2, "
                    "e = document.elementFromPoint(cx, cy); return e === elem;",
                    button
                )
                if not is_visible:
                    logger.info("Nút 'Xem thêm' bị che, xử lý phần tử che phủ...")
                    driver.execute_script(
                        """
                        var elements = document.querySelectorAll(
                            'div[role="dialog"], div[aria-label="Tìm kiếm"], input.x1i10hfl.xggy1nq, '
                            'div[data-visualcompletion="loading"], div[role="banner"], div[aria-label="Messenger"]'
                        );
                        elements.forEach(function(el) { el.style.display = 'none'; });
                        """
                    )
                    time.sleep(1)
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", button)
                    time.sleep(0.5)

                WebDriverWait(driver, 10).until(EC.element_to_be_clickable(button))
                button.click()
                logger.info("Đã nhấn nút 'Xem thêm' bằng Selenium.")
                time.sleep(5)
                return True
            except Exception as e:
                logger.warning(f"Lỗi khi nhấn nút Xem thêm (Selenium): {str(e)}")
                try:
                    driver.execute_script("arguments[0].click();", button)
                    logger.info("Đã nhấn nút 'Xem thêm' bằng JavaScript.")
                    time.sleep(5)
                    return True
                except Exception as js_e:
                    logger.warning(f"Lỗi khi nhấn nút Xem thêm (JavaScript): {str(js_e)}")
                    continue
        return False
    except Exception as e:
        logger.warning(f"Lỗi khi tìm nút Xem thêm: {str(e)}")
        return False

# Hàm cuộn trang để tải bài viết
def scroll_to_load_posts(driver, num_posts):
    logger.info("Cuộn trang để tải bài viết...")
    scroll_height = driver.execute_script("return document.body.scrollHeight")
    current_height = 0
    posts_loaded = 0
    max_scrolls = max(num_posts * 2, 20)
    scroll_count = 0

    while posts_loaded < num_posts and scroll_count < max_scrolls:
        driver.execute_script(f"window.scrollTo(0, {current_height + 1500});")
        time.sleep(3)
        posts_loaded = len(driver.find_elements(By.CSS_SELECTOR, "div.x1yztbdb.x1n2onr6.xh8yej3.x1ja2u2z"))
        current_height += 1500
        scroll_count += 1
        new_scroll_height = driver.execute_script("return document.body.scrollHeight")
        if current_height >= new_scroll_height:
            break
    logger.info(f"Đã tải {posts_loaded} bài viết sau {scroll_count} lần cuộn.")

# Hàm chính để crawl bài viết
def crawl_and_save(num_posts: int, fanpage_link: str):
    try:
        if num_posts <= 0:
            raise ValueError("Số lượng bài viết phải lớn hơn 0")
        if not fanpage_link:
            raise ValueError("Link fanpage không được để trống")

        # Thiết lập ChromeDriver
        chrome_options = Options()
        chrome_options.add_argument("--disable-notifications")
        # chrome_options.add_argument("--headless")  # Bỏ comment nếu muốn chạy ẩn
        driver = webdriver.Chrome(options=chrome_options)

        # Đăng nhập vào Facebook
        driver.get("https://www.facebook.com")
        time.sleep(random.uniform(2, 4))

        try:
            email_field = driver.find_element(By.ID, "email")
            logger.info("Chưa đăng nhập, tiến hành đăng nhập...")
            human_like_typing(email_field, FACEBOOK_EMAIL)
            pass_field = driver.find_element(By.ID, "pass")
            human_like_typing(pass_field, FACEBOOK_PASSWORD)
            login_button = driver.find_element(By.NAME, "login")
            login_button.click()
            time.sleep(random.uniform(3, 5))

            try:
                captcha_input = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "captcha"))
                )
                logger.warning("Phát hiện CAPTCHA! Vui lòng nhập thủ công...")
                input("Nhấn Enter sau khi nhập CAPTCHA: ")
            except Exception:
                logger.info("Không phát hiện CAPTCHA.")

            pickle.dump(driver.get_cookies(), open("facebook_cookies.pkl", "wb"))
            logger.info("Đã lưu cookie sau đăng nhập.")
        except Exception as e:
            logger.info(f"Có thể đã đăng nhập hoặc lỗi: {str(e)}")

        # Truy cập fanpage
        driver.get(fanpage_link)
        time.sleep(5)

        # Xử lý thông báo
        try:
            not_now_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'Không bây giờ') or contains(text(), 'Not Now')]"))
            )
            not_now_button.click()
            logger.info("Đã từ chối thông báo trên fanpage.")
        except Exception:
            logger.info("Không có thông báo trên fanpage.")

        # Cuộn để tải bài viết
        scroll_to_load_posts(driver, num_posts)

        # Lấy danh sách bài viết
        post_elements = driver.find_elements(By.CSS_SELECTOR, "div.x1yztbdb.x1n2onr6.xh8yej3.x1ja2u2z")
        logger.info(f"Tìm thấy {len(post_elements)} bài viết.")

        posts = []
        valid_post_count = 0
        conn = connect_to_db()

        for index, post_element in enumerate(post_elements[:num_posts], 1):
            try:
                driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", post_element)
                time.sleep(8)

                # Nhấn nút "Xem thêm"
                click_see_more_in_post(driver, post_element)

                # Lấy HTML của bài viết
                post_html = post_element.get_attribute("outerHTML")
                post_soup = BeautifulSoup(post_html, "html.parser")

                # Lấy URL bài viết
                url_elements = post_soup.find_all("a", class_="x1i10hfl xjbqb8w x1ejq31n xd10rxx x1sy0etr x17r0tee x972fbf xcfux6l x1qhh985 xm0m39n x9f619 x1ypdohk xt0psk2 xe8uvvx xdj266r x11i5rnm xat24cr x1mh8g0r xexx8yu x4uap5 x18d9i69 xkhd6sd x16tdsg8 x1hl2dhg xggy1nq x1a2a7pz x1heor9g x1lku1pv")
                url = url_elements[0]['href'] if url_elements else f"post_{index}_no_url"

                # Lấy nội dung bài viết
                content_divs = post_soup.find_all("div", dir="auto")
                seen_texts = set()
                full_text_lines = []
                for div in content_divs:
                    text = div.get_text(strip=True)
                    if text and text not in seen_texts:
                        seen_texts.add(text)
                        full_text_lines.append(text)
                full_text = "\n".join(full_text_lines)

                if full_text:
                    # Xử lý #MeoF hoặc @MeoF
                    if "#MeoF" in full_text:
                        last_meof_index = full_text.rfind("#MeoF")
                        full_text = full_text[:last_meof_index].strip()
                    elif "@MeoF" in full_text:
                        last_meof_index = full_text.rfind("@MeoF")
                        full_text = full_text[:last_meof_index].strip()

                    valid_post_count += 1
                    logger.info(f"Bài viết {valid_post_count}: {full_text[:50]}... (URL: {url})")

                    # Tách bài viết thành các sub_posts
                    sub_posts = []
                    matches = list(re.finditer(r'(?m)^(#\d+)\s*?\n', full_text))
                    if len(matches) < 2:
                        if full_text.strip():
                            sub_posts.append({"text": full_text})
                    else:
                        second_hash_start = matches[1].start()
                        first_post_text = full_text[:second_hash_start].strip()
                        if first_post_text:
                            sub_posts.append({"text": first_post_text})

                        remaining_text = full_text[second_hash_start:]
                        segments = re.split(r'(?m)^(#\d+)\s*?\n', remaining_text)
                        if segments[0].strip() == "":
                            segments = segments[1:]

                        for i in range(0, len(segments), 2):
                            if i + 1 >= len(segments):
                                break
                            segment_number = segments[i]
                            segment_text = segments[i + 1].strip()
                            if segment_text:
                                sub_posts.append({"text": f"{segment_number}\n{segment_text}"})

                    # Lưu vào database
                    for sub_post in sub_posts:
                        save_to_posts(conn, url, sub_post["text"])

                    posts.append({"url": url, "sub_posts": sub_posts})
                else:
                    logger.warning(f"Bài viết {index} không có nội dung.")
            except Exception as e:
                logger.error(f"Lỗi khi xử lý bài viết {index}: {str(e)}")
                continue

        conn.close()
        driver.quit()
        if len(posts) < num_posts:
            logger.warning(f"Chỉ crawl được {len(posts)}/{num_posts} bài viết.")
        return posts

    except Exception as e:
        logger.error(f"Lỗi trong crawl_and_save: {str(e)}")
        if 'driver' in locals():
            driver.quit()
        raise

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python crawler.py <num_posts> <fanpage_link>")
        sys.exit(1)
    try:
        num_posts = int(sys.argv[1])
        fanpage_link = sys.argv[2]
        crawled_posts = crawl_and_save(num_posts, fanpage_link)
        print(json.dumps(crawled_posts))
    except ValueError as e:
        print(f"Lỗi: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi khi chạy crawler: {str(e)}", file=sys.stderr)
        sys.exit(1)