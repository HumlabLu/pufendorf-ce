use iced::widget::text::LineHeight;
use iced::widget::operation::snap_to;
use iced::widget::scrollable::RelativeOffset;
use iced::widget::Id;
use log::{debug, error, info, trace, LevelFilter};
use flexi_logger::{DeferredNow, Record};
use flexi_logger::{Duplicate, FileSpec, LogSpecification, Logger, WriteMode};
use iced::{
    widget::{button, column, container, row, scrollable, slider, text, text_input},
    Element, Length, Settings, Task, Theme,
};
use std::{
    sync::{Arc, LazyLock, Mutex},
};
use async_stream::stream;
use tokio_stream::StreamExt;
use serde_json::Value;
use std::fs;
use std::path::Path;
use openai_dive::v1::api::Client;
use openai_dive::v1::endpoints::chat::RoleTrackingStream;
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
use lance::{create_database, append_documents, get_row_count};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use arrow_schema::{DataType, Field, Schema};
use arrow_array::{
    RecordBatch
};
use lancedb::query::QueryBase;
use lancedb::query::ExecutableQuery;
use iced::futures::TryStreamExt;
use arrow_array::{Float32Array, StringArray, Array};
use tokio::runtime::Runtime;

mod structs;
use structs::*;

mod embedder;
use embedder::chunk_file_txt;

// LOG is the Id for the chat log output pane, needed in the snap_to(...) function.
static LOG: LazyLock<Id> = LazyLock::new(|| Id::new("log"));

#[derive(Parser)]
#[command(version, about, long_about = "Reading data.")]
struct Cli {
    /// Sets the level of logging;
    /// error, warn, info, debug, or trace
    #[arg(short, long, default_value = "info")]
    log_level: String,

    #[arg(short, long, help = "Append text file with info.")]
    append: Option<String>,

    #[arg(short, long, help = "Retrieval cut off.", default_value_t = 1.0)]
    cutoff: f32,

    #[arg(short, long, help = "DB name.")]
    dbname: Option<String>,

    #[arg(short, long, help = "Text file with info.")]
    filename: Option<String>,

    #[arg(long, help = "Font size.", default_value_t = 18)]
    fontsize: u32,

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
        now.format("%Y-%m-%d %H:%M:%S"),
        record.level(),
        file_name,
        line,
        &record.args()
    )
}


fn theme(_: &App) -> Theme {
    Theme::TokyoNight //GruvboxLight //Dark
}


async fn connect_db(db_name: String) -> lancedb::Result<lancedb::Connection> {
    lancedb::connect(&db_name).execute().await
}

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
        .duplicate_to_stderr(Duplicate::Info) // was ::All
        .write_mode(WriteMode::BufferAndFlush)
        .start().expect("Logging?");
    info!("Start");
    
    // ------------ New Db stuff
    
    // Embedding model (downloads once, then runs locally).
    // See also embedder.rs
    let mut embedder = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
    ).expect("No embedding model.");

    // Embedder, plus determine dimension.
    let model_info = TextEmbedding::get_model_info(&EmbeddingModel::AllMiniLML6V2);
    let dim = model_info.unwrap().dim as i32;
    info!("Embedding dim {}", dim);

    let one_embeddings = embedder.embed(&["one"], None).expect("Cannot embed?");

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

    if let Some(ref filename) = cli.filename {
        info!("Filename {filename}.");
        /*
        let chunks = chunk_file_txt(filename, 512);
        if let Ok(lines) = chunks {
            for line in lines {
                println!("{:?}", line);
            }
        }*/
        let rt = Runtime::new().unwrap();
        let _ = rt.block_on(create_database(filename));
    }

    if let Some(ref filename) = cli.append{
        info!("Append filename {filename}.");
        let rt = Runtime::new().unwrap();
        let _ = rt.block_on(append_documents(filename));
    }

    let rt = Runtime::new().unwrap();
    let _db: Option<lancedb::Connection> = match rt.block_on(connect_db(db_name)) {
        Ok(db) => Some(db),
        Err(_e) => {
            error!("DB Error!");
            None
        }
    };
    
    info!("Row count: {}", rt.block_on(get_row_count("docs")));
    
    // Have DB connexion here?
    let config = AppConfig {
        db_path: "data/lancedb_fastembed".into(),
        model: "gpt-4o-mini".into(),
        fontsize: cli.fontsize,
        cut_off: cli.cutoff,
    };

    iced::application(
        move || App::new(config.clone()),
        App::update,
        App::view,
        )
        // iced::application(App::new, App::update, App::view)
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
    fn new(config: AppConfig) -> (Self, Task<Message>) {
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

        // Alternative way? which is best?
        let rt = Runtime::new().unwrap();
        let dbc: Option<lancedb::Connection> = match rt.block_on(connect_db(config.db_path.clone())) {
            Ok(db) => Some(db),
            Err(e) => {
                error!("DB Error! {e}");
                None
            }
        };
        // Probably does not have to be an Arc/Mutex.
        let db_connexion = Arc::new(Mutex::new(dbc));

        (Self {
            config, 

            // gpt-5-nano
            model: "gpt-4.1-nano".into(),
            mode: Mode::Chat,

            temperature: 0.05,
            num_predict: 1024,

            // Draft is user input, lines are the text in the output pane.
            draft: String::new(),
            lines: vec![Line {
                role: Role::System,
                content: "".into(),
            }],
            waiting: false,

            history,

            system_prompt: sysprompt,
            extra_info: "The year is 1667".into(), // Not used.

            db_connexion: db_connexion,
            
        }, Task::none())
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

            Message::CutOffChanged(t) => {
                self.config.cut_off = t;
                Task::none()
            }

            Message::NumPredictChanged(n) => {
                self.num_predict = n;
                Task::none()
            }

            Message::ResetParams => {
                self.temperature = 0.1;
                self.num_predict = 1024;
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
                        Task::stream(
                            stream_chat_oai(model, prompt, opts, self.history.clone(), self.db_connexion.clone(), self.config.cut_off)
                        )
                    }
                };
                // Task::none()
                Task::batch([task, snap_to(LOG.clone(), RelativeOffset::END)])
            }

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
        // const MY_SIZE: u32 = 20;

        // info!("{:?}", self.config.db_path);
        
        let transcript = self.lines.iter().fold(column![].spacing(8), |col, line| {
            let prefix = match line.role {
                Role::User => "You: ",
                Role::Assistant => "Samuel: ",
                Role::System => "",
            };
            col.push(text(format!("{prefix}{}", line.content)).size(self.config.fontsize).font(MY_FONT).line_height(LineHeight::Relative(1.4)))
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
            text(format!("T: {:.2}", self.temperature)).font(MY_FONT),
            slider(0.0..=1.2, self.temperature, Message::TemperatureChanged)
                .width(Length::FillPortion(2))
                .step(0.05),
            text(format!("CO: {:.1}", self.config.cut_off)).font(MY_FONT),
            slider(0.0..=2.0, self.config.cut_off, Message::CutOffChanged)
                .width(Length::FillPortion(2))
                .step(0.1),
            text(format!("Max tokens: {}", self.num_predict)).font(MY_FONT),
            slider(1..=4096, self.num_predict, Message::NumPredictChanged)
                .width(Length::FillPortion(1))
                .step(12),
            // button(text("Reset").font(MY_FONT)).on_press(Message::ResetParams),
            button(text("Clear").font(MY_FONT)).on_press(Message::ClearAll),
        ]
        .spacing(12).align_y(iced::Alignment::Center).padding(10);

        let input = text_input("Ask your question…", &self.draft)
            .on_input(Message::DraftChanged)
            .on_submit(Message::Submit)
            .padding(10)
            .size(self.config.fontsize)
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
    dbc: Arc<Mutex<Option<lancedb::Connection>>>,
    cut_off: f32,
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
        let _flagged = result.results[0].flagged;
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
            InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                .with_show_download_progress(true),
        ).expect("No embedding model.");
        let table_name = "docs".to_string();
        let dim = 384;
        let db: lancedb::Connection = {
            let guard = dbc.lock().unwrap();
            guard.clone().take().expect("Expected a database connection!")
        };

        let _schema = Arc::new(Schema::new(vec![
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
                .refine_factor(4)
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
                    let text = if texts.is_null(i) { "<NULL>" } else { texts.value(i) };
                    if dist < cut_off {
                        debug!("{dist:.3} * {text}");
                        context += text;
                    } else {
                        debug!("{dist:.3}   {text}");
                    }
                }
            }
        };

        let mut messages: Vec<ChatMessage> = {
            let h = history.lock().unwrap();
            h.iter().map(line_to_chat_message).collect()
        };

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

        { // Inside a block because we cannot hold the guard.
            let mut h = history.lock().unwrap();
            info!("Q: {}", user_prompt);
            h.push(Line { role: Role::User, content: user_prompt });
            info!("A: {}", assistant_acc);
            h.push(Line { role: Role::Assistant, content: assistant_acc });
        }

        yield Message::LlmDone;
    }
}
