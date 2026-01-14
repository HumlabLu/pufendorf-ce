
# Intro

Speak with Samuel von Pufendorf. Simple GUI which streams OpenAI output into a text pane.

# Usage

## Prepare Database

The database can be created with simple tab-separated text data. The first item is meta information (book, chapter, page, etc), and the second item is the data that will be stored as a vector for retrieval.

Here is an example line from biographical data, from the point of view of Samuel himself.
```shell
Encyclopedia/Introduction 	 Your approach was secular, non-metaphysical, and anti-authoritarian; it eschewed religious appeals, scholastic dogma, essentialism, teleology, and the frequent mix of these that appealed to many German thinkers, Catholic and Protestant alike.
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

Simple RAG system using the OpenAI API. Uses `lancedb` database.

