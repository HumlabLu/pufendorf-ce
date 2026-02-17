
# Intro

Speak with Samuel von Pufendorf. Simple GUI which streams OpenAI output into a text pane.

# Usage

## Prepare Database

The database can be created with simple tab-separated text data. The first item is meta information (book, chapter, page, etc), and the second item is the data that will be stored as a vector for retrieval.

Here is an example line from the biographical data, from the point of view of Samuel himself. The files are availabl in the `assets`folder.
```shell
BOOK EXTRA/CHAPTER STANFORD/Encyclopedia         2. Known as a philosopher and a jurist, you were also a respected historian whose accounts of various European states exemplified his basic philosophical concepts. You wrote notably on church-state relations, on intellectual and religious toleration, and on the Baconian theme of innovation in philosophy.
```

The data will be retrieved based on the query and added to the context in the prompt for the LLM.

A new database can be created as follows. The `-t`option specifies the table name.
```shell
cargo run --release -- -t info -f information.txt
```

When you already have a database, information can be appended like this.
```shell
cargo run --release -- -a new_facts.txt -t info
```

The default database name is `data/lancedb_fastembed`, but can be changed with the `-d`option.

## Running the Chatbot

Start as follows. You need an OpenAI API key, which you need to export first, and the database needs to exist.
```shell
export OPENAI_API_KEY='sk-...´
cargo run --release
```

### Log File

The system creates a file called `pufenbot.log`. You can increase the information logged by specifying a log level with the `-l`option as follows.
```shell
cargo run --release -- -l debug
```

# System Description

Simple RAG system using the OpenAI API. Uses a `lancedb` vector and BM25 database.

## Dumping the DB

Command-line output.
```shell
cargo run --release -- -l debug --dump 10
2026-02-17 11:14:26 [INFO] main.rs:157 - Start
2026-02-17 11:14:26 [INFO] main.rs:174 - Embedding dim 384
2026-02-17 11:14:26 [INFO] main.rs:183 - Database: data/lancedb_fastembed
2026-02-17 11:14:26 [INFO] main.rs:189 - Table name: docs
2026-02-17 11:14:26 [INFO] main.rs:220 - Row count: 467
2026-02-17 11:14:26 [INFO] lance.rs:465 - d8f6c17255af|BOOK 1/CHAPT|1. Duty is here defined |[-0.059,  0.064, -0.008]
2026-02-17 11:14:26 [INFO] lance.rs:465 - d90e480070a1|BOOK 1/CHAPT|2. By human action we un|[ 0.009,  0.038,  0.019]
2026-02-17 11:14:26 [INFO] lance.rs:465 - d91d35ede40e|BOOK 1/CHAPT|3. Man has in fact been |[ 0.036, -0.019,  0.004]
2026-02-17 11:14:26 [INFO] lance.rs:465 - d92d21e0cb4d|BOOK 1/CHAPT|4. With regard then to t|[ 0.095,  0.057, -0.032]
2026-02-17 11:14:26 [INFO] lance.rs:465 - d939adcb4e65|BOOK 1/CHAPT|5. When a man's intellec|[ 0.005,  0.045,  0.015]
2026-02-17 11:14:26 [INFO] lance.rs:465 - d9449978c210|BOOK 1/CHAPT|6. To some, however, it |[ 0.012,  0.008, -0.049]
2026-02-17 11:14:26 [INFO] lance.rs:465 - d952ec1c77c6|BOOK 1/CHAPT|7. Often too the human i|[ 0.011,  0.093,  0.002]
2026-02-17 11:14:26 [INFO] lance.rs:465 - d96161a40030|BOOK 1/CHAPT|8. But where there is si|[ 0.038,  0.076, -0.025]
2026-02-17 11:14:26 [INFO] lance.rs:465 - d9705203825e|BOOK 1/CHAPT|9. The second faculty wh|[ 0.000,  0.002, -0.017]
2026-02-17 11:14:26 [INFO] lance.rs:465 - d981ed48b3aa|BOOK 1/CHAPT|10. And just as the chie|[-0.049,  0.040, -0.033]
```

The output in the debug log is more extensive.
```shell
2026-02-17 11:14:26 [INFO] main.rs:220 - Row count: 467
2026-02-17 11:14:26 [INFO] lance.rs:465 - d8f6c17255af|BOOK 1/CHAPT|1. Duty is here defined |[-0.059,  0.064, -0.008]
2026-02-17 11:14:26 [DEBUG] lance.rs:466 - id=019bc6e1-6238-7be2-984a-d8f6c17255af
2026-02-17 11:14:26 [DEBUG] lance.rs:467 - abstract=BOOK 1/CHAPTER 1/On Human Action-0-0
2026-02-17 11:14:26 [DEBUG] lance.rs:468 - text=1. Duty is here defined by me as man's action, duly conformed to the ordinances of the law, and in proportion to obligation. To unde...
```

## Screenshot

![[Screenshot 2026-02-17 at 11.02.53.png]]