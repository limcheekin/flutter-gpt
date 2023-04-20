-- Create a function to search for documents
create function match_documents(embedding vector(768), match_count int)
returns table(uuid uuid, document text, cmetadata json, custom_id text, similarity float)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    uuid,
    document,
    cmetadata,
    custom_id,
    1 - (documents.embedding <=> query_embedding) as similarity
  from langchain_pg_embedding documents
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$
;

DROP TABLE messages CASCADE;
DROP TABLE down_vote_reasons CASCADE;
DROP TABLE conversations CASCADE;
DROP TABLE users CASCADE;
DROP TABLE channels CASCADE;
DROP TABLE models CASCADE;

CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL, 
  password VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE models (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) UNIQUE NOT NULL,
  max_tokens INTEGER NOT NULL,
  cmetadata JSON 
);

CREATE TABLE channels (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) UNIQUE NOT NULL,
  cmetadata JSON 
);

CREATE TABLE conversations (
  id BIGSERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id) NOT NULL,
  channel_id INTEGER REFERENCES channels(id) NOT NULL,
  context TEXT NOT NULL,
  cmetadata JSON,
  created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX conversation_user_id_idx ON conversations(user_id);
CREATE INDEX conversation_channel_id_idx ON conversations(channel_id);

CREATE TABLE down_vote_reasons (
  id SERIAL PRIMARY KEY,
  description VARCHAR(255) UNIQUE NOT NULL 
);

CREATE TABLE messages (
  id BIGSERIAL PRIMARY KEY,
  conversation_id BIGINT REFERENCES conversations(id) NOT NULL,
  user_id INTEGER REFERENCES users(id) NOT NULL, 
  user_message TEXT NOT NULL,
  bot_message TEXT NOT NULL,
  vote SMALLINT DEFAULT 0, -- 0: neutral, 1: thumb_up, -1: thumb_down
  down_vote_reason_id INTEGER REFERENCES down_vote_reasons(id), 
  user_feedback TEXT,
  model_id INTEGER REFERENCES models(id) NOT NULL, 
  cmetadata JSON,
  created_at TIMESTAMP DEFAULT NOW() 
);
CREATE INDEX message_conversation_id_idx ON messages(conversation_id); 
CREATE INDEX message_user_id_idx ON messages(user_id);
