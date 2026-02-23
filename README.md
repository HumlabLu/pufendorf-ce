
# Intro

Speak with Samuel von Pufendorf. Simple GUI which streams OpenAI output into a text pane.

The first version of the Pufendorf chatbot was developed for the 'What is lost//What is found' DigiJustice exhibition at the Pufendorf institude in May 2025 [DigiJustice](https://www.pi.lu.se/en/themes-0/digijustice-rethinking-digital-inequalities-and-human-rights-age-ai). That version was also displayed at 'Kulturnatten' in September 2025. This version is a complete rewrite in Rust.

# Usage

## Running the Chatbot

Start as follows. You need an OpenAI API key, which you need to export first, and the database needs to exist (see below).
```shell
export OPENAI_API_KEY='sk-...´
cargo run --release
```

It is also possible to use Ollama instead of OpenAI, but this feature is still experimental. Specify `-m ollama` to change to ollama, and specify the local model with `-M`. This also assumes that Ollama is running locally on the default port.

Example.
```shell
cargo run --release -- -m ollama -M "llama3.2:latest"
```

### Sliders

There are three sliders above the input field, controlling temperature, cutoff point and context size.

 - 'T' controls the temperature. A higher temperature lets the LLM generate more freely, whilst a lower temperature keeps the output more grounded to the question and the retrieved data.
 - 'CO' controls the cutoff for the retrieved items from the vector database. Each item has a value; a lower value tends to mean the retrieved item is more relevant to the question.
 - 'Max CTX' controls how many retrieved items will be added to the prompt. 

The 'CO' slider only has effect in the OpenAI plus vector retrieval mode (the default).

## Prepare a Database

The database can be created with simple tab-separated text data. The first item is meta information (book, chapter, page, etc), and the second item is the data that will be stored as a vector for retrieval.

The default database name is `data/lancedb_fastembed`, but can be changed with the `-d` option. The default table name is `docs`, which can be changed with the `-t` option.

Here is an example line from the biographical data, from the point of view of Samuel himself. The files are available in the `assets`folder.

```shell
BOOK EXTRA/CHAPTER STANFORD/Encyclopedia         2. Known as a philosopher and a jurist, you were also a respected historian whose accounts of various European states exemplified his basic philosophical concepts. You wrote notably on church-state relations, on intellectual and religious toleration, and on the Baconian theme of innovation in philosophy.
```

The data will be retrieved based on the query and added to the context in the prompt for the LLM.

A new table with information in the database can be created as follows. The `-t`option specifies the table name.
```shell
cargo run --release -- -t info -f information.txt
```

When you already have a database, information can be appended like this.
```shell
cargo run --release -- -t info -a new_facts.txt 
```

## Vector and Full-Text Retrieval

The `-s` option selects between vector, full-text or both database retrieval methods. The default is vector only. When both are selected, context is retrieved using both vector and full-text searches. The results are combined with a re-ranker. 

## Log File

The system creates a file called `pufenbot.log`. You can increase the information logged by specifying a log level with the `-l`option as follows.
```shell
cargo run --release -- -l debug
```

Without specifying a log-level, some info about the database and the queries and answers are printed to the terminal.
```shell
2026-02-19 08:49:02 [INFO] main.rs:229 - Start
2026-02-19 08:49:02 [INFO] main.rs:273 - Embedding dim 384
2026-02-19 08:49:02 [INFO] main.rs:282 - Database: data/lancedb_fastembed
2026-02-19 08:49:02 [INFO] main.rs:288 - Table name: docs
2026-02-19 08:49:02 [INFO] main.rs:319 - Row count: 467
2026-02-19 08:49:13 [INFO] main.rs:1406 - Q: Who are you?
2026-02-19 08:49:14 [INFO] main.rs:1501 - Retrieved 1 (1.47-1.66) items.
2026-02-19 08:49:16 [INFO] main.rs:1577 - A: I am Samuel Von Pufendorf, a humble servant of learning and jurisprudence, born in Saxony in the year of our Lord 1632. My pursuits have been manifold: I have endeavoured to elucidate the natural law that underpins civil society, to examine the relations between church and state, and to promote the principles of toleration and peace among nations. My writings extend also into the realm of history, wherein I have sought to record the deeds of European states, not merely as chronicles but as reflections of the fundamental philosophical principles that govern human conduct and societal organisation. My work is thus a synthesis of philosophy, law, and history, aimed at fostering understanding and order within the human community.
```

# Prompt

The assets directory contains a file called `chatprompts.json`, containing the base prompt for the system. It also contains a field (an array) for extra hard-coded information. The label is displayed in the text output pane.

Example chatprompts file.
````json
{
  "system_prompt":"You are Samuel Von Pufendorf. Answer in the style of a 17th century academic.",
  "extra_info":["You were born: January 8, 1632, Dorfchemnitz, near Thalheim, Saxony."],
  "label":"Samuel: "
}
```

# Font and Fontsize

The default font is Fira, but you can choose another font with the `--fontname` option. The size can be changed with the `--fontsize` option.
 
```shell
cargo run --release -- --fontname Sarasa --fontsize 24
```

# System Description

Simple RAG system using the OpenAI API. Uses a `lancedb` vector and BM25 database.

```shell
Usage: pufendorf-ce [OPTIONS]

Options:
  -l, --log-level <LOG_LEVEL>    Sets the level of logging; error, warn, info, debug, or trace [default: info]
  -a, --append <APPEND>          Append text file with info.
      --chunksize <CHUNKSIZE>    Chunk size. [default: 4096]
  -c, --cutoff <CUTOFF>          Retrieval cut off. [default: 1.5]
  -d, --dbname <DBNAME>          DB name.
  -e, --embedmodel <EMBEDMODEL>  Embedding model. [default: all-minilm-l6-v2]
      --dump <DUMP>              Dump DB contents. [default: 0]
  -f, --filename <FILENAME>      Text file with info.
      --fontsize <FONTSIZE>      Font size. [default: 18]
      --fontname <FONTNAME>      Font name. [default: Fira]
  -m, --mode <MODE>              Mode, openai or ollama. [default: openai]
  -s, --searchmode <SEARCHMODE>  Search mode, vector, fulltext or both. [default: vector]
  -M, --model <MODEL>            Model name [default: gpt-4.1-nano]
  -p, --promptfile <PROMPTFILE>  System prompt/info json file.
  -t, --tablename <TABLENAME>    Table name.
  -T, --themestr <THEMESTR>      Theme. [default: tokyonight]
  -h, --help                     Print help (see more with '--help')
  -V, --version                  Print version
```
  
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

## Build Notes

On a mac, xcode and homebrew are needed to compile the chatbot.

```shell
xcode-select --install
brew install protobuf
cargo build --release
```

On an 8GB MacBook Air (Intel, 13-inch, 2020) it takes about 75 minutes to build. It takes 11 minutes on an M1 MacBook Pro (16-inch, 2021).

## Screenshot

[<img alt="Screenshot" src="assets/Screenshot 2026-02-17 at 11.02.53.png" />](Screenshot)
