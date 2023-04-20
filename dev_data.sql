INSERT INTO users (name, email, password) VALUES ('limcheekin', 'limcheekin@email.com', 'nopassword');
INSERT INTO channels (name) VALUES ('android');
INSERT INTO models (name, max_tokens) VALUES ('google/flan-t5-base', 512);
INSERT INTO down_vote_reasons (description) 
                       VALUES ('Incorrect information'), 
                              ('Missing information'), 
                              ('Lack of understanding'), 
                              ('Repetitive'), 
                              ('Other');
