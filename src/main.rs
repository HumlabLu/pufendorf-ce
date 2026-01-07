use iced::widget::text::LineHeight;
use iced::widget::operation::snap_to;
use iced::widget::scrollable::RelativeOffset;
use iced::widget::Id;
use log::{debug, error, info, trace, LevelFilter};
use flexi_logger::{DeferredNow, Record};
use flexi_logger::{Duplicate, FileSpec, LogSpecification, Logger, WriteMode};
use iced::{
    widget::{button, column, container, pick_list, row, scrollable, slider, text, text_input},
    Element, Length, Settings, Task, Theme,
};
use std::{
    fmt,
    sync::{Arc, LazyLock, Mutex},
};
use async_stream::stream;
use tokio_stream::StreamExt;
use bm25::{Document, Language, SearchEngineBuilder, SearchResult};
use serde_json::Value;
use std::fs;
use std::path::Path;
use openai_dive::v1::api::Client;
use openai_dive::v1::endpoints::chat::RoleTrackingStream;
use openai_dive::v1::models::Gpt5Model;
use openai_dive::v1::resources::chat::{
    ChatCompletionParametersBuilder, ChatCompletionResponseFormat, ChatMessage, ChatMessageContent,
    DeltaChatMessage,
};
use openai_dive::v1::models::ModerationModel;
use openai_dive::v1::resources::moderation::{ModerationInput, ModerationParametersBuilder};
use clap::Parser;
use std::io::Write;
use std::str::FromStr;

mod lance;
use lance::{read_file_to_vec, append_documents};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use arrow_schema::{DataType, Field, Schema};
use arrow_array::{
    types::Float32Type, FixedSizeListArray, Int32Array, RecordBatch, RecordBatchIterator,
};
use lancedb::query::QueryBase;
use lancedb::query::ExecutableQuery;
use iced::futures::TryStreamExt;
use arrow_array::{Float32Array, StringArray, Array};

// LOG is the Id for the chat log output pane, needed in the snap_to(...) function.
static LOG: LazyLock<Id> = LazyLock::new(|| Id::new("log"));

#[derive(Parser)]
#[command(version, about, long_about = "Reading data.")]
struct Cli {
    /// Sets the level of logging;
    /// error, warn, info, debug, or trace
    #[arg(short, long, default_value = "info")]
    log_level: String,

    #[arg(short, long, help = "DB name.")]
    dbname: Option<String>,

    #[arg(short, long, help = "Table name.")]
    tablename: Option<String>,
}

fn log_format(
    w: &mut dyn Write,
    now: &mut DeferredNow,
    record: &Record,
) -> Result<(), std::io::Error> {
    let file_path = record.file().unwrap_or("<unknown>");
    let file_name = Path::new(file_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("<unknown>");
    let line = record.line().unwrap_or(0);
    write!(
        w,
        "{} [{}] {}:{} - {}",
        now.format("%Y-%m-%d %H:%M:%S"), // Format without standard timezone.
        record.level(),
        file_name,
        line,
        &record.args()
    )
}


fn theme(_: &App) -> Theme {
    Theme::TokyoNight //GruvboxLight //Dark
}

// ----
// Struct for model options like temp &c.
#[derive(Debug, Default, Clone)]
pub struct ModelOptions {
    pub temperature: f32,
    pub num_predict: i32,
}
impl ModelOptions {
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn num_predict(mut self, num_predict: i32) -> Self {
        self.num_predict = num_predict;
        self
    }
}
// ----

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Chat,
}
impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Mode::Chat => write!(f, "Chat (history)"),
        }
    }
}

#[derive(Debug, Clone)]
enum Role {
    User,
    Assistant,
    System,
}
#[derive(Debug, Clone)]
struct Line {
    role: Role,
    content: String,
}

// Messages for iced GUI.
#[derive(Debug, Clone)]
enum Message {
    DraftChanged(String),
    Submit,
    ModeChanged(Mode),

    TemperatureChanged(f32),
    NumPredictChanged(i32),
    MaxTurnsChanged(u16),

    ResetParams,
    ClearAll,

    LlmChunk(String),
    LlmDone,
    LlmErr(String),
}

// "Global" data for the iced app.
struct App {
    model: String,
    mode: Mode,

    temperature: f32,
    num_predict: i32,
    max_turns: u16,

    draft: String, // User input
    lines: Vec<Line>,
    waiting: bool,

    history: Arc<Mutex<Vec<Line>>>,

    system_prompt: String,
    extra_info: String,

    db: Option<lancedb::Connection>,
}


// #[tokio::main]
async fn _openai_stream() {
    let client = Client::new_from_env();

    let parameters = ChatCompletionParametersBuilder::default()
        .model(Gpt5Model::Gpt5Nano.to_string())
        .messages(vec![
            ChatMessage::User {
                content: ChatMessageContent::Text("Hello!".to_string()),
                name: None,
            },
            ChatMessage::User {
                content: ChatMessageContent::Text(
                    "What are the five biggest cities in Vietnam?".to_string(),
                ),
                name: None,
            },
        ])
        .response_format(ChatCompletionResponseFormat::Text)
        .build()
        .unwrap();

    let stream = client.chat().create_stream(parameters).await.unwrap();

    // The stream will receive a chunk of a chat completion response. The first chunk will contain the role of the message, and subsequent chunks won't contain the role.
    // When a chunk is received without a role, it will use the `DeltaChatMessage::Untagged` variant. And you'll have to manually infer the role of the message based on the previous messages.
    // The 'RoleTrackingStream' is a wrapper around the stream that does this for you - it will automatically infer the role of the message and return the correct variant.

    let mut tracked_stream = RoleTrackingStream::new(stream);

    while let Some(response) = tracked_stream.next().await {
        match response {
            Ok(chat_response) => {
                chat_response
                    .choices
                    .iter()
                    .for_each(|choice| match &choice.delta {
                        DeltaChatMessage::User { content, .. } => {
                            print!("{content}");
                        }
                        DeltaChatMessage::System { content, .. } => {
                            print!("{content}");
                        }
                        DeltaChatMessage::Assistant {
                            content: Some(chat_message_content),
                            ..
                        } => {
                            print!("{chat_message_content}");
                        }
                        _ => {}
                    })
            }
            Err(e) => eprintln!("{e}"),
        }
    }
}

async fn connect_db(db_name: String) -> lancedb::Result<lancedb::Connection> {
    lancedb::connect(&db_name).execute().await
}

// #[tokio::main]
fn main() -> iced::Result {
    let cli = Cli::parse();

    // This switches off logging from html5 and other crates.
    let level_filter = LevelFilter::from_str(&cli.log_level).unwrap_or(LevelFilter::Off);
    let log_spec = LogSpecification::builder()
        .module("html5ever", LevelFilter::Off)
        .module("rusty_puff", level_filter) // Sets our level to the one on the cli.
        .build();

    let _logger = Logger::with(log_spec)
        .format(log_format)
        .log_to_file(
            FileSpec::default()
                .suppress_timestamp()
                .basename("pufenbot")
                .suffix("log"),
        )
        .append()
        .duplicate_to_stderr(Duplicate::All)
        .write_mode(WriteMode::BufferAndFlush)
        .start().expect("Logging?");
    info!("Start");
    
    // ------------ New Db stuff
    
    // Embedding model (downloads once, then runs locally).
    let mut embedder = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
    ).expect("No embedding model.");

    // Embedder, plus determine dimension.
    let one_embeddings = embedder.embed(&["one"], None).expect("Cannot embed?");
    let dim = one_embeddings[0].len() as i32;
    info!("Embedding dim {}", dim);

    // My DB. Created if it doesn't exist.
    let db_name = match cli.dbname {
        Some(dbn) => dbn,
        None => "data/lancedb_fastembed".to_string()
    };
    info!("Database: {db_name}");

    let table_name = match cli.tablename {
        Some(tbln) => tbln,
        None => "docs".to_string()
    };
    info!("Table name: {table_name}");

    // code moved to streaming function.
    
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("abstract", DataType::Utf8, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
            false,
        ),
    ]));

    // ------------

    /*
    let file_path = Path::new("chatprompts.json");
    // Read the entire content of the JSON file into a string
    let content = fs::read_to_string(file_path).expect("no file");
    // Print just to confirm file reading success
    // println!("File Content:\n{}", content);
    let data: Value = serde_json::from_str(&content).expect("data");
    // println!("{}", &data);
    let sysprompt = &data["system_prompt"]
        .as_str()
        .unwrap_or("You are Samuel Von Pufendorf.");
    let mut sysprompt = sysprompt.to_string();
    // dbg!(sysprompt);
    if let Some(extras) = data["extra_info"].as_array() {
        for extra in extras {
            // println!("{}", extra);
            sysprompt += extra.as_str().unwrap_or("");
        }
    }
    */
    // get_models();
    // openai_stream();

    let corpus = [
        "The rabbit munched the orange carrot.",
        "The snake hugged the green lizard.",
        "The hedgehog impaled the orange orange.",
        "The squirrel buried the brown nut.",
    ];

    let search_engine = SearchEngineBuilder::<u32>::with_corpus(Language::English, corpus).build();

    let limit = 3;
    let search_results = search_engine.search("orange", limit);
    debug!("{:?}", search_results);
    let search_results = search_engine.search("When were you born?", limit);
    debug!("{:?}", search_results);

    iced::application(App::new, App::update, App::view)
        .title("Speak with Pufendorf")
        .theme(theme)
        .settings(Settings::default())
        .font(include_bytes!("../assets/FiraMono-Medium.ttf").as_slice())
        /*
        .settings(Settings {
            default_text_size: 24.into(),
            ..Settings::default()
        })*/
        .run()
}

impl App {
    fn new() -> Self {
        // Read the prompts from a json file.
        // Should contain a system_prompt and extra_info.
        let file_path = Path::new("assets/chatprompts.json");
        let content = fs::read_to_string(file_path).expect("no file");
        let data: Value = serde_json::from_str(&content).expect("data");
        let sysprompt = &data["system_prompt"]
            .as_str()
            .unwrap_or("You are Samuel Von Pufendorf.");
        let mut sysprompt = sysprompt.to_string();
        if let Some(extras) = data["extra_info"].as_array() {
            sysprompt += "\n";
            for extra in extras {
                sysprompt += "\n";
                sysprompt += extra.as_str().unwrap_or("");
            }
        }
        debug!("{}", sysprompt);

        let history = Arc::new(Mutex::new(vec![
            Line {
                role: Role::System,
                content: sysprompt.clone(),
            }]
        ));

        Self {
            // gpt-5-nano
            model: "gpt-4.1-nano".into(), //Gpt5Model::Gpt5Nano.to_string()
            mode: Mode::Chat,

            temperature: 0.05,
            num_predict: 1024,
            max_turns: 20,

            // Draft is user input, lines are everyting in the output pane.
            draft: String::new(),
            lines: vec![Line {
                role: Role::System,
                content: "".into(),
            }],
            waiting: false,

            history,

            system_prompt: sysprompt,
            extra_info: "The year is 1667".into(), // Not used.

            db: None,
        }
    }

    fn update(&mut self, msg: Message) -> Task<Message> {
        match msg {
            Message::DraftChanged(s) => {
                self.draft = s;
                Task::none()
            }

            Message::ModeChanged(m) => {
                self.mode = m;
                Task::none()
            }

            Message::TemperatureChanged(t) => {
                self.temperature = t;
                Task::none()
            }

            Message::NumPredictChanged(n) => {
                self.num_predict = n;
                Task::none()
            }

            Message::MaxTurnsChanged(n) => {
                self.max_turns = n;
                Task::none()
            }

            Message::ResetParams => {
                self.temperature = 0.7;
                self.num_predict = 512;
                self.max_turns = 20;
                Task::none()
            }

            Message::ClearAll => {
                self.draft.clear();
                self.waiting = false;
                self.lines = vec![Line {
                    role: Role::System,
                    content: "".into(),
                }];
                *self.history.lock().unwrap() = vec![Line {
                    role: Role::System,
                    content: self.system_prompt.clone(),
                    //"You are a helpful assistant.".to_string(),
                }];
                Task::none()
            }

            Message::Submit => {
                if self.waiting {
                    return Task::none();
                }
                let prompt = self.draft.trim().to_string();
                if prompt.is_empty() {
                    return Task::none();
                }

                self.draft.clear();
                self.waiting = true;

                self.lines.push(Line {
                    role: Role::User,
                    content: prompt.clone(),
                });
                self.lines.push(Line {
                    role: Role::Assistant,
                    content: String::new(),
                });

                let model = self.model.clone();
                let opts = ModelOptions::default()
                    .temperature(self.temperature)
                    .num_predict(self.num_predict);

                let task = match self.mode {
                    Mode::Chat => {
                        Task::stream(stream_chat_oai(model, prompt, opts, self.history.clone()))
                    }
                };
                // Task::none()
                Task::batch([task, snap_to(LOG.clone(), RelativeOffset::END)])
            }

            /*Message::LlmChunk(chunk) => {
                if let Some(last) = self.lines.last_mut() {
                    if matches!(last.role, Role::Assistant) {
                        last.content.push_str(&chunk);
                    }
                }
                snap_to(LOG.clone(), RelativeOffset::END)
            }*/
            
            // Add to the last (added as empty) Line in self.lines.
            Message::LlmChunk(chunk) => {
                if let Some(last) = self.lines.last_mut() {
                    if matches!(last.role, Role::Assistant) {
                        trace!("{}", chunk);
                        if chunk.starts_with(&last.content) {
                            last.content = chunk;
                        } else {
                            last.content.push_str(&chunk);
                        }
                    }
                }
                snap_to(LOG.clone(), RelativeOffset::END)
            }

            Message::LlmDone => {
                self.waiting = false;
                if self.mode == Mode::Chat {
                    // Here we used to trim.
                    if let Some(last) = self.lines.last_mut() {
                        last.content.push_str("\n");
                    }
                    trace!("H: {:?}", self.history);
                }
                snap_to(LOG.clone(), RelativeOffset::END)
            }

            Message::LlmErr(e) => {
                self.waiting = false;
                if let Some(last) = self.lines.last_mut() {
                    if matches!(last.role, Role::Assistant) && last.content.is_empty() {
                        last.content = format!("(error) {e}");
                        return Task::none();
                    }
                }
                self.lines.push(Line {
                    role: Role::System,
                    content: format!("(error) {e}"),
                });
                Task::none()
            }
        }
    }

    fn view(&self) -> Element<'_, Message> {
        // const MY_FONT: iced::Font = iced::Font::with_name("JetBrainsMonoNL NFM");
        const MY_FONT: iced::Font = iced::Font::with_name("FiraMono Nerd Font Mono");
        const MY_SIZE:u32 = 20;

        let transcript = self.lines.iter().fold(column![].spacing(8), |col, line| {
            let prefix = match line.role {
                Role::User => "You: ",
                Role::Assistant => "Samuel: ",
                Role::System => "",
            };
            col.push(text(format!("{prefix}{}", line.content)).size(MY_SIZE).font(MY_FONT).line_height(LineHeight::Relative(1.4)))
        });

        let top = container(
            scrollable(container(transcript).padding(12).width(Length::Fill))
                .id(LOG.clone())
                .height(Length::Fill),
        )
        .width(Length::Fill)
        .height(Length::Fill);

        let controls = row![
            // text("Mode:").font(MY_FONT),
            // pick_list(&MODES[..], Some(self.mode), Message::ModeChanged),
            text(format!("Temp: {:.1}", self.temperature)).font(MY_FONT),
            slider(0.0..=2.0, self.temperature, Message::TemperatureChanged)
                .width(Length::FillPortion(2))
                .step(0.05),
            text(format!("Max tokens: {}", self.num_predict)).font(MY_FONT),
            slider(1..=4096, self.num_predict, Message::NumPredictChanged)
                .width(Length::FillPortion(1))
                .step(12),
            // text(format!("Max turns: {}", self.max_turns)),
            // slider(1u16..=100u16, self.max_turns, Message::MaxTurnsChanged)
                // .width(Length::Fixed(160.0)),
            // button(text("Reset").font(MY_FONT)).on_press(Message::ResetParams),
            button(text("Clear").font(MY_FONT)).on_press(Message::ClearAll),
        ]
        .spacing(12).align_y(iced::Alignment::Center).padding(10);

        let input = text_input("Type your question…", &self.draft)
            .on_input(Message::DraftChanged)
            .on_submit(Message::Submit)
            .padding(10)
            .size(MY_SIZE)
            .font(MY_FONT)
            .width(Length::Fill);

        let bottom = container(
            column![
                controls,
                row![input, text(if self.waiting { "  thinking…" } else { "" })].spacing(8),
            ]
            .spacing(8),
        )
        .width(Length::Fill)
        .padding(8);

        column![top, bottom]
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }
}
/*
fn stream_chat(
    model: String,
    user_prompt: String,
    opts: ModelOptions,
    history: Arc<Mutex<Vec<ChatMessage>>>,
) -> impl tokio_stream::Stream<Item = Message> + Send + 'static {
    stream! {
        let ollama = Ollama::default();
        let req = ChatMessageRequest::new(model, vec![ChatMessage::user(user_prompt)]).options(opts);

        let mut s = match ollama.send_chat_messages_with_history_stream(history, req).await {
            Ok(s) => s,
            Err(e) => { yield Message::LlmErr(e.to_string()); yield Message::LlmDone; return; }
        };

        while let Some(item) = s.next().await {
            match item {
                Ok(res) => {
                    let chunk = res.message.content;
                    if !chunk.is_empty() { yield Message::LlmChunk(chunk); }
                }
                Err(_) => { yield Message::LlmErr("Ollama stream error".into()); yield Message::LlmDone; return; }
            }
        }

        yield Message::LlmDone;
    }
}
*/
fn _stream_chat(
    _model: String,
    _user_prompt: String,
    _opts: ModelOptions,
    _history: Arc<Mutex<Vec<Line>>>,
) -> impl tokio_stream::Stream<Item = Message> + Send + 'static {
    stream! {

        let txt = vec!["Some", "words"];
        for t in txt {
            let chunk = t;
            if !chunk.is_empty() {
                yield Message::LlmChunk(chunk.to_string());
                yield Message::LlmChunk(" ".to_string());
            }
        }

        yield Message::LlmDone;
    }
}


// Function to convert Line to ChatMessage.
// TODO let history use ChatMessage instead.
fn line_to_chat_message(line: &Line) -> ChatMessage {
    let content = ChatMessageContent::Text(line.content.clone());
    // println!("{}", content);
    match line.role {
        Role::User => ChatMessage::User { content, name: None },
        Role::System => ChatMessage::System { content, name: None },
        Role::Assistant => ChatMessage::Assistant {
            content: Some(content),
            reasoning_content: None,
            refusal: None,
            name: None, //Some("Pufendorf".to_string()), // TODO check?
            audio: None,
            tool_calls: None,
        },
    }
}

fn stream_chat_oai(
    model: String,
    user_prompt: String,
    opts: ModelOptions,
    history: Arc<Mutex<Vec<Line>>>,
) -> impl tokio_stream::Stream<Item = Message> + Send + 'static {
    stream! {
        let client = Client::new_from_env();

        // Moderation ----
        let parameters = ModerationParametersBuilder::default()
                .model(ModerationModel::OmniModerationLatest.to_string())
                // .input(ModerationInput::Text("I want to kill them.".to_string()))
                .input(ModerationInput::Array(vec![
                    user_prompt.clone()
                ]))
                .build()
                .unwrap();
        let result = client.moderations().create(parameters).await.unwrap();
        let cats = &result.results[0].category_scores;
        let flagged = result.results[0].flagged;
        // println!("Mod: {:?}", cats);
        
        // The moderator is very strict, we check scores.
        let mut flagged = false;
        let v = serde_json::to_value(&cats).unwrap();
        if let Value::Object(map) = v {
            for (k, v) in map {
                let score = v.as_f64().unwrap();
                if score > 0.6 {
                    debug!("{k}: {score}");
                    flagged = true;
                }
            }
        }
        
        if flagged {
            debug!("Mod: {:?}", cats);
            yield Message::LlmChunk("Please ask another question!".to_string());
            yield Message::LlmDone;
            return;
        }


        debug!("Searching context.");
        let mut context = "Use the following info to answer the question, if there is none, use your own knowledge.\n".to_string();
        
        // insert Db/RAG here?
        let mut embedder = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
        ).expect("No embedding model.");
        let db_name = "data/lancedb_fastembed";
        let table_name = "docs".to_string();
        let dim = 384;
        let db = lancedb::connect(&db_name).execute().await.expect("Cannot connect to DB.");
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("abstract", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                false,
            ),
        ]));
        let q = embedder.embed(vec![&user_prompt], None).expect("err");
        let qv = &q[0];

        if let Ok(ref table) = db.open_table(&table_name).execute().await {
            let results: Vec<RecordBatch> = table
                .query()
                .nearest_to(qv.as_slice()).expect("err")
                .limit(12)
                .execute()
                .await.expect("err")
                .try_collect()
                .await.expect("err");
            for b in &results {
                let text_idx = b.schema().index_of("text").expect("err");
                let dist_idx = b.schema().index_of("_distance").expect("err");

                let texts = b.column(text_idx).as_any().downcast_ref::<StringArray>().unwrap();
                let dists = b.column(dist_idx).as_any().downcast_ref::<Float32Array>().unwrap();

                for i in 0..b.num_rows() {
                    let dist = dists.value(i);
                    if dist < 1.0 {
                        let text = if texts.is_null(i) { "<NULL>" } else { texts.value(i) };
                        debug!("{dist:.3}  {text}");
                        context += text;
                    }
                }
            }
        };

        let mut messages: Vec<ChatMessage> = {
            let h = history.lock().unwrap();
            h.iter().map(line_to_chat_message).collect()
        };
        // println!("{:?}", messages);

        // messages.push(context_message);
        messages.push(ChatMessage::User {
            content: ChatMessageContent::Text(context),
            name: None,
        });
        
        messages.push(ChatMessage::User {
            content: ChatMessageContent::Text(user_prompt.clone()),
            name: None,
        });
        debug!("{:?}", messages);

        let max_tokens = u32::try_from(opts.num_predict).ok();
        let params = match ChatCompletionParametersBuilder::default()
            .model(model)
            .messages(messages)
            .max_tokens(max_tokens.unwrap())
            .temperature(opts.temperature)
            .response_format(ChatCompletionResponseFormat::Text)
            .build()
        {
            Ok(p) => p,
            Err(e) => {
                yield Message::LlmErr(e.to_string());
                yield Message::LlmDone; return;
            }
        };

        let stream0 = match client.chat().create_stream(params).await {
            Ok(s) => s,
            Err(e) => { yield Message::LlmErr(e.to_string()); yield Message::LlmDone; return; }
        };

        let mut tracked = RoleTrackingStream::new(stream0);

        let mut assistant_acc = String::new();

        while let Some(item) = tracked.next().await {
            match item {
                Ok(chat_response) => {
                    for choice in &chat_response.choices {
                        if let DeltaChatMessage::Assistant { content: Some(delta), .. } = &choice.delta {
                            let s = delta.to_string();
                            if !s.is_empty() {
                                assistant_acc.push_str(&s);
                                yield Message::LlmChunk(s);
                            }
                        }
                    }
                }
                Err(e) => { yield Message::LlmErr(e.to_string()); yield Message::LlmDone; return; }
            }
        }

        {
            let mut h = history.lock().unwrap();
            info!("Q: {}", user_prompt);
            h.push(Line { role: Role::User, content: user_prompt });
            // println!("Pusing: {}", &assistant_acc);
            info!("A: {}", assistant_acc);
            h.push(Line { role: Role::Assistant, content: assistant_acc });
        }

        yield Message::LlmDone;
    }
}
