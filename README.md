
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

