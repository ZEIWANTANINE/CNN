CREATE TABLE crawler_posts (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    url VARCHAR(255) NOT NULL UNIQUE,
    text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE crawler_posts DROP CONSTRAINT crawler_posts_url_key;

select * from crawler_posts;
