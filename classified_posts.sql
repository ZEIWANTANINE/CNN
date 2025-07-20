CREATE TABLE classified_posts (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    url VARCHAR(255) NOT NULL,
    text TEXT,
    neutral BOOLEAN,
    negative BOOLEAN,
    positive BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

select * from classified_posts;