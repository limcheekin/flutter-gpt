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

CREATE TABLE users (
  user_id SERIAL PRIMARY KEY,
  name VARCHAR(255),
  email VARCHAR(255) UNIQUE, 
  password VARCHAR(255),
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE channels (
  channel_id SERIAL PRIMARY KEY,
  name VARCHAR(255) UNIQUE 
);

CREATE TABLE conversations (
  conversation_id BIGSERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(user_id),
  channel_id INTEGER REFERENCES channels(channel_id),
  context TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX conversation_user_id_idx ON conversations(user_id);
CREATE INDEX conversation_channel_id_idx ON conversations(channel_id);

CREATE TABLE down_vote_reasons (
  reason_id SERIAL PRIMARY KEY,
  description VARCHAR(255) UNIQUE 
);

CREATE TABLE messages (
  message_id BIGSERIAL PRIMARY KEY,
  conversation_id INTEGER REFERENCES conversations(conversation_id),
  user_id INTEGER REFERENCES users(user_id), 
  user_message TEXT,
  bot_message TEXT,
  vote smallint, -- 0: neutral, 1: thumb_up, -1: thumb_down
  down_vote_reason_id INTEGER REFERENCES down_vote_reasons(reason_id), 
  user_feedback TEXT,
  created_at TIMESTAMP DEFAULT NOW() 
);
CREATE INDEX message_conversation_id_idx ON messages(conversation_id); 
CREATE INDEX message_user_id_idx ON messages(user_id);
